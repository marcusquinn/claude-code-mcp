#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
  type ServerResult,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve as pathResolve } from 'node:path';
import * as path from 'path';
import { readFileSync } from 'node:fs';

// Server version - update this when releasing new versions
const SERVER_VERSION = "1.10.12";

// Define debugMode globally using const
const debugMode = process.env.MCP_CLAUDE_DEBUG === 'true';

// Track if this is the first tool use for version printing
let isFirstToolUse = true;

// Capture server startup time when the module loads
const serverStartupTime = new Date().toISOString();

// Dedicated debug logging function
export function debugLog(message?: any, ...optionalParams: any[]): void {
  if (debugMode) {
    console.error(message, ...optionalParams);
  }
}

/**
 * Message format for conversation context
 */
interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Extended interface for Claude Code tool arguments with session support
 */
interface ClaudeCodeArgs {
  prompt: string;
  workFolder?: string;
  sessionId?: string;
  messages?: ConversationMessage[];
  stateless?: boolean;
}

/**
 * Claude CLI JSON output format
 */
interface ClaudeCliResponse {
  type: string;
  subtype: string;
  is_error: boolean;
  duration_ms: number;
  duration_api_ms?: number;
  num_turns?: number;
  result: string;
  session_id: string;
  total_cost_usd?: number;
  usage?: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
}

/**
 * Session mapping: parentSessionId -> claudeCliSessionId
 * Limited to MAX_SESSIONS to prevent memory leaks in long-running servers
 */
const MAX_SESSIONS = 1000;
const sessionMap = new Map<string, string>();

/**
 * Store a session mapping with LRU-style eviction
 */
function setSessionMapping(parentId: string, claudeId: string): void {
  if (sessionMap.size >= MAX_SESSIONS) {
    // Remove oldest entry (first inserted)
    const firstKey = sessionMap.keys().next().value;
    if (firstKey) sessionMap.delete(firstKey);
  }
  sessionMap.set(parentId, claudeId);
}

/**
 * Determines the Claude CLI command/path based on the following precedence:
 * 1. An absolute path specified in the `CLAUDE_CLI_NAME` environment variable.
 * 2. The local user installation at `~/.claude/local/claude`.
 * 3. A simple command name from `CLAUDE_CLI_NAME` (looked up in the system's PATH).
 * 4. The default command `claude` (looked up in the system's PATH).
 *
 * Note: Relative paths in `CLAUDE_CLI_NAME` are not allowed and will cause an error.
 */
export function findClaudeCli(): string {
  debugLog('[Debug] Attempting to find Claude CLI...');

  const customCliName = process.env.CLAUDE_CLI_NAME;
  if (customCliName) {
    debugLog(`[Debug] Using custom Claude CLI name from CLAUDE_CLI_NAME: ${customCliName}`);
    
    if (path.isAbsolute(customCliName)) {
      debugLog(`[Debug] CLAUDE_CLI_NAME is an absolute path: ${customCliName}`);
      return customCliName;
    }
    
    if (customCliName.startsWith('./') || customCliName.startsWith('../') || customCliName.includes('/')) {
      throw new Error(`Invalid CLAUDE_CLI_NAME: Relative paths are not allowed. Use either a simple name (e.g., 'claude') or an absolute path (e.g., '/tmp/claude-test')`);
    }
  }
  
  const cliName = customCliName || 'claude';

  const userPath = join(homedir(), '.claude', 'local', 'claude');
  debugLog(`[Debug] Checking for Claude CLI at local user path: ${userPath}`);

  if (existsSync(userPath)) {
    debugLog(`[Debug] Found Claude CLI at local user path: ${userPath}. Using this path.`);
    return userPath;
  } else {
    debugLog(`[Debug] Claude CLI not found at local user path: ${userPath}.`);
  }

  debugLog(`[Debug] Falling back to "${cliName}" command name, relying on spawn/PATH lookup.`);
  console.warn(`[Warning] Claude CLI not found at ~/.claude/local/claude. Falling back to "${cliName}" in PATH. Ensure it is installed and accessible.`);
  return cliName;
}

/**
 * Format conversation messages into a context string for Claude CLI
 */
function formatConversationContext(messages: ConversationMessage[]): string {
  if (!messages || messages.length === 0) return '';
  
  const formatted = messages.map(msg => {
    const roleLabel = msg.role === 'user' ? 'User' : 'Assistant';
    return `[${roleLabel}]: ${msg.content}`;
  }).join('\n\n');
  
  return `<conversation_context>\n${formatted}\n</conversation_context>\n\n`;
}

/**
 * Translate slash commands to @ mentions for Claude Code subagent invocation
 * Only matches /command patterns that don't look like file paths (no / after the command name)
 */
function translateSlashCommands(prompt: string): string {
  // Match /command at start of line, but only if followed by space, end of line, or end of string
  // This avoids matching file paths like /tmp/foo or /usr/bin/something
  return prompt.replace(/^\/([a-zA-Z][a-zA-Z0-9_-]*)(?=\s|$)/gm, '@$1');
}

/**
 * Parse Claude CLI JSON response
 * Handles both clean JSON output and output with extra text
 */
function parseClaudeResponse(stdout: string): ClaudeCliResponse | null {
  try {
    // Try parsing the entire stdout first (most common case with --output-format json)
    const trimmed = stdout.trim();
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      const parsed = JSON.parse(trimmed);
      if (parsed.type === 'result') return parsed;
    }
    
    // Fallback: try to find a JSON object line by line
    const lines = stdout.split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith('{') && line.endsWith('}')) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.type === 'result') return parsed;
        } catch {
          // Continue to next line
        }
      }
    }
    
    return null;
  } catch (e) {
    debugLog('[Debug] Failed to parse Claude CLI JSON response:', e);
    return null;
  }
}

export async function spawnAsync(command: string, args: string[], options?: { timeout?: number, cwd?: string }): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    debugLog(`[Spawn] Running command: ${command} ${args.join(' ')}`);
    const process = spawn(command, args, {
      shell: false,
      timeout: options?.timeout,
      cwd: options?.cwd,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    process.stdout.on('data', (data) => { stdout += data.toString(); });
    process.stderr.on('data', (data) => {
      stderr += data.toString();
      debugLog(`[Spawn Stderr Chunk] ${data.toString()}`);
    });

    process.on('error', (error: NodeJS.ErrnoException) => {
      debugLog(`[Spawn Error Event] Full error object:`, error);
      let errorMessage = `Spawn error: ${error.message}`;
      if (error.path) {
        errorMessage += ` | Path: ${error.path}`;
      }
      if (error.syscall) {
        errorMessage += ` | Syscall: ${error.syscall}`;
      }
      errorMessage += `\nStderr: ${stderr.trim()}`;
      reject(new Error(errorMessage));
    });

    process.on('close', (code) => {
      debugLog(`[Spawn Close] Exit code: ${code}`);
      debugLog(`[Spawn Stderr Full] ${stderr.trim()}`);
      debugLog(`[Spawn Stdout Full] ${stdout.trim()}`);
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`Command failed with exit code ${code}\nStderr: ${stderr.trim()}\nStdout: ${stdout.trim()}`));
      }
    });
  });
}

/**
 * MCP Server for Claude Code with Session Continuity
 */
export class ClaudeCodeServer {
  private server: Server;
  private claudeCliPath: string;
  private packageVersion: string;

  constructor() {
    this.claudeCliPath = findClaudeCli();
    console.error(`[Setup] Using Claude CLI command/path: ${this.claudeCliPath}`);
    this.packageVersion = SERVER_VERSION;

    this.server = new Server(
      {
        name: 'claude_code',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();

    this.server.onerror = (error) => console.error('[Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers(): void {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'claude_code',
          description: `Claude Code Agent with Session Continuity: Multi-modal assistant for code, file, Git, and terminal operations via Claude CLI.

**Session Continuity** (Default behavior):
- Pass \`sessionId\` from your parent interface to maintain conversation context
- First call: optionally include \`messages\` array with conversation history
- Subsequent calls: just pass \`sessionId\` and \`prompt\` - context is maintained
- Response includes session ID for tracking

**Subagent Invocation**:
- Use \`/commandname\` in prompts to invoke Claude Code subagents (translated to @mentions)

**Stateless Mode** (Legacy):
- Set \`stateless: true\` for single-prompt behavior without session tracking

**Capabilities**:
• File ops: Create, read, edit, move, copy, delete, list, analyze images
• Code: Generate, analyze, refactor, fix bugs
• Git: Stage, commit, push, tag, create PRs
• Terminal: Run CLI commands, open URLs
• Web search + summarization
• Multi-step workflows

**Tips**:
1. Be concise and explicit for complex tasks
2. Use \`workFolder\` for file operations
3. For long tasks, split into smaller steps
4. Combine operations in sequences for efficiency
`,
          inputSchema: {
            type: 'object',
            properties: {
              prompt: {
                type: 'string',
                description: 'The message or prompt for Claude to process.',
              },
              workFolder: {
                type: 'string',
                description: 'Working directory for file operations. Must be an absolute path.',
              },
              sessionId: {
                type: 'string',
                description: 'Session ID from the parent interface. Enables conversation continuity - subsequent calls with the same sessionId will resume the Claude Code session.',
              },
              messages: {
                type: 'array',
                description: 'Full conversation history from the parent interface. Only needed on the first call - subsequent calls use sessionId for continuity.',
                items: {
                  type: 'object',
                  properties: {
                    role: {
                      type: 'string',
                      enum: ['user', 'assistant'],
                      description: 'The role of the message sender.',
                    },
                    content: {
                      type: 'string',
                      description: 'The message content.',
                    },
                  },
                  required: ['role', 'content'],
                },
              },
              stateless: {
                type: 'boolean',
                description: 'Force single-prompt mode without session continuity. Default is false (sessions enabled).',
                default: false,
              },
            },
            required: ['prompt'],
          },
        }
      ],
    }));

    const executionTimeoutMs = 1800000; // 30 minutes timeout

    this.server.setRequestHandler(CallToolRequestSchema, async (args, call): Promise<ServerResult> => {
      debugLog('[Debug] Handling CallToolRequest:', args);

      const toolName = args.params.name;
      if (toolName !== 'claude_code') {
        throw new McpError(ErrorCode.MethodNotFound, `Tool ${toolName} not found`);
      }

      const toolArguments = args.params.arguments;
      let prompt: string;

      if (
        toolArguments &&
        typeof toolArguments === 'object' &&
        'prompt' in toolArguments &&
        typeof toolArguments.prompt === 'string'
      ) {
        prompt = toolArguments.prompt;
      } else {
        throw new McpError(ErrorCode.InvalidParams, 'Missing or invalid required parameter: "prompt" must be a string.');
      }

      // Validate optional sessionId
      const sessionId = toolArguments.sessionId;
      if (sessionId !== undefined && typeof sessionId !== 'string') {
        throw new McpError(ErrorCode.InvalidParams, 'Invalid parameter: "sessionId" must be a string if provided.');
      }

      // Validate optional messages array
      const messages = toolArguments.messages;
      if (messages !== undefined) {
        if (!Array.isArray(messages)) {
          throw new McpError(ErrorCode.InvalidParams, 'Invalid parameter: "messages" must be an array if provided.');
        }
        for (const msg of messages) {
          if (typeof msg !== 'object' || msg === null || 
              typeof msg.role !== 'string' || typeof msg.content !== 'string' ||
              (msg.role !== 'user' && msg.role !== 'assistant')) {
            throw new McpError(ErrorCode.InvalidParams, 'Invalid parameter: each message must have "role" (user|assistant) and "content" (string).');
          }
        }
      }

      let effectiveCwd = homedir();

      if (toolArguments.workFolder && typeof toolArguments.workFolder === 'string') {
        const resolvedCwd = pathResolve(toolArguments.workFolder);
        debugLog(`[Debug] Specified workFolder: ${toolArguments.workFolder}, Resolved to: ${resolvedCwd}`);

        if (existsSync(resolvedCwd)) {
          effectiveCwd = resolvedCwd;
          debugLog(`[Debug] Using workFolder as CWD: ${effectiveCwd}`);
        } else {
          debugLog(`[Warning] Specified workFolder does not exist: ${resolvedCwd}. Using default: ${effectiveCwd}`);
        }
      } else {
        debugLog(`[Debug] No workFolder provided, using default CWD: ${effectiveCwd}`);
      }

      try {
        debugLog(`[Debug] Attempting to execute Claude CLI with prompt: "${prompt}" in CWD: "${effectiveCwd}"`);

        if (isFirstToolUse) {
          const versionInfo = `claude_code v${SERVER_VERSION} started at ${serverStartupTime}`;
          console.error(versionInfo);
          isFirstToolUse = false;
        }

        // Use already-validated values
        const parentSessionId = sessionId;
        const validatedMessages = messages as ConversationMessage[] | undefined;
        const stateless = toolArguments.stateless === true;

        let processedPrompt = translateSlashCommands(prompt);

        const claudeProcessArgs: string[] = ['--dangerously-skip-permissions'];

        if (!stateless && parentSessionId) {
          const existingClaudeSessionId = sessionMap.get(parentSessionId);
          
          if (existingClaudeSessionId) {
            debugLog(`[Debug] Resuming Claude CLI session: ${existingClaudeSessionId} for parent session: ${parentSessionId}`);
            claudeProcessArgs.push('--resume', existingClaudeSessionId);
          } else if (validatedMessages && validatedMessages.length > 0) {
            debugLog(`[Debug] First call for parent session: ${parentSessionId}, injecting ${validatedMessages.length} messages as context`);
            const contextPrefix = formatConversationContext(validatedMessages);
            processedPrompt = contextPrefix + 'Continue the conversation. ' + processedPrompt;
          }
        }

        claudeProcessArgs.push('--output-format', 'json');
        claudeProcessArgs.push('-p', processedPrompt);

        debugLog(`[Debug] Invoking Claude CLI: ${this.claudeCliPath} ${claudeProcessArgs.join(' ')}`);

        const { stdout, stderr } = await spawnAsync(
          this.claudeCliPath,
          claudeProcessArgs,
          { timeout: executionTimeoutMs, cwd: effectiveCwd }
        );

        debugLog('[Debug] Claude CLI stdout:', stdout.trim());
        if (stderr) {
          debugLog('[Debug] Claude CLI stderr:', stderr.trim());
        }

        const parsedResponse = parseClaudeResponse(stdout);
        
        let resultText: string;
        let claudeSessionId: string | undefined;

        if (parsedResponse) {
          resultText = parsedResponse.result;
          claudeSessionId = parsedResponse.session_id;
          
          if (!stateless && parentSessionId && claudeSessionId) {
            setSessionMapping(parentSessionId, claudeSessionId);
            debugLog(`[Debug] Stored session mapping: ${parentSessionId} -> ${claudeSessionId}`);
          }

          if (parsedResponse.usage) {
            debugLog(`[Debug] Token usage - Input: ${parsedResponse.usage.input_tokens}, Output: ${parsedResponse.usage.output_tokens}`);
          }
          if (parsedResponse.total_cost_usd !== undefined) {
            debugLog(`[Debug] Cost: $${parsedResponse.total_cost_usd.toFixed(4)}`);
          }
        } else {
          resultText = stdout;
          debugLog('[Debug] Could not parse JSON response, using raw stdout');
        }

        const responseContent: { type: 'text'; text: string }[] = [
          { type: 'text', text: resultText }
        ];

        if (!stateless && claudeSessionId) {
          responseContent.push({
            type: 'text',
            text: `\n---\n_Session ID: ${claudeSessionId}_`
          });
        }

        return { content: responseContent };

      } catch (error: any) {
        debugLog('[Error] Error executing Claude CLI:', error);
        let errorMessage = error.message || 'Unknown error';
        if (error.stderr) {
          errorMessage += `\nStderr: ${error.stderr}`;
        }
        if (error.stdout) {
          errorMessage += `\nStdout: ${error.stdout}`;
        }

        if (error.signal === 'SIGTERM' || (error.message && error.message.includes('ETIMEDOUT')) || (error.code === 'ETIMEDOUT')) {
          throw new McpError(ErrorCode.InternalError, `Claude CLI command timed out after ${executionTimeoutMs / 1000}s. Details: ${errorMessage}`);
        }
        throw new McpError(ErrorCode.InternalError, `Claude CLI execution failed: ${errorMessage}`);
      }
    });
  }

  async run(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Claude Code MCP server running on stdio');
  }
}

const server = new ClaudeCodeServer();
server.run().catch(console.error);
