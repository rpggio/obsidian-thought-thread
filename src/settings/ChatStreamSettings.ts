import { CHAT_MODELS, DEFAULT_OPENAI_COMPLETIONS_URL } from 'src/openai/chatGPT'

export interface ChatStreamSettings {
	/**
	 * The API key to use when making requests
	 */
	apiKey: string

	/**
	 * The URL endpoint for chat
	 */
	apiUrl: string

	/**
	 * The GPT model to use
	 */
	apiModel: string

	/**
	 * Custom model name when using customize option
	 */
	customModelName: string

	/**
	 * The temperature to use when generating responses (0-2). 0 means no randomness.
	 */
	temperature: number

	/**
	 * The system prompt sent with each request to the API
	 */
	systemPrompt: string

	/**
	 * Enable debug output in the console
	 */
	debug: boolean

	/**
	 * The maximum number of tokens to send (up to model limit). 0 means as many as possible.
	 */
	maxInputTokens: number

	/**
	 * The maximum number of tokens to return from the API. 0 means no limit. (A token is about 4 characters).
	 */
	maxResponseTokens: number

	/**
	 * The maximum depth of ancestor notes to include. 0 means no limit.
	 */
	maxDepth: number
}

export const DEFAULT_SYSTEM_PROMPT = `
You are a critical-thinking assistant bot. 
Consider the intent of my questions before responding.
Do not restate my information unless I ask for it. 
Do not include caveats or disclaimers.
Use step-by-step reasoning. Be brief.
`.trim()

export const DEFAULT_SETTINGS: ChatStreamSettings = {
	apiKey: '',
	apiUrl: DEFAULT_OPENAI_COMPLETIONS_URL,
	apiModel: CHAT_MODELS.GPT_35_TURBO.name,
	customModelName: '',
	temperature: 1,
	systemPrompt: DEFAULT_SYSTEM_PROMPT,
	debug: false,
	maxInputTokens: 0,
	maxResponseTokens: 0,
	maxDepth: 0
}

export function getModels() {
	return Object.entries(CHAT_MODELS).map(([, value]) => value.name)
}
