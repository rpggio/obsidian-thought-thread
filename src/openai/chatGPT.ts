import { request, RequestUrlParam } from 'obsidian'
import { openai } from './chatGPT-types'

export const DEFAULT_OPENAI_COMPLETIONS_URL = `https://api.openai.com/v1/chat/completions`

export const getOpenAICompletionsURL = (configuredUrl?: string): string => {
	return configuredUrl || DEFAULT_OPENAI_COMPLETIONS_URL
}

export type ChatModelSettings = {
	name: string,
	tokenLimit: number,
	encodingFrom?: string
}

export const CHAT_MODELS = {
	GPT_35_TURBO: {
		name: 'gpt-3.5-turbo',
		tokenLimit: 4096
	},
	GPT_35_TURBO_0125: {
		name: 'gpt-3.5-turbo-0125',
		tokenLimit: 16385
	},
	GPT_35_16K: {
		name: 'gpt-3.5-turbo-16k',
		tokenLimit: 16385
	},
	GPT_35_TURBO_1106: {
		name: 'gpt-3.5-turbo-1106',
		tokenLimit: 16385
	},
	GPT_4o: {
		name: 'gpt-4o',
		tokenLimit: 128000
	},
	GPT_4o_MINI: {
		name: 'gpt-4o-mini',
		encodingFrom: 'gpt-4o',
		tokenLimit: 16384
	},
	GPT_4: {
		name: 'gpt-4',
		tokenLimit: 8192
	},
	GPT_4_TURBO_PREVIEW: {
		name: 'gpt-4-turbo-preview',
		tokenLimit: 128000
	},
	GPT_4_0125_PREVIEW: {
		name: 'gpt-4-0125-preview',
		tokenLimit: 128000
	},
	GPT_4_1106_PREVIEW: {
		name: 'gpt-4-1106-preview',
		tokenLimit: 128000
	},
	GPT_4_0613: {
		name: 'gpt-4-0613',
		tokenLimit: 8192
	},
	GPT_4_32K: {
		name: 'gpt-4-32k',
		tokenLimit: 32768
	},
	GPT_4_32K_0613: {
		name: 'gpt-4-32k-0613',
		tokenLimit: 32768
	},
	CUSTOMIZE: {
		name: 'customize',
		tokenLimit: 0
	}
}

export type ChatGPTModel = keyof typeof CHAT_MODELS

export function chatModelByName(name: string) {
	return Object.values(CHAT_MODELS).find((model) => model.name === name)
}

export const defaultChatGPTSettings: Partial<openai.CreateChatCompletionRequest> =
{
	model: CHAT_MODELS.GPT_35_TURBO.name,
	max_tokens: 500,
	temperature: 0,
	top_p: 1.0,
	presence_penalty: 0,
	frequency_penalty: 0,
	stop: []
}

export async function getChatGPTCompletion(
	apiKey: string,
	apiUrl: string,
	model: openai.CreateChatCompletionRequest['model'],
	messages: openai.CreateChatCompletionRequest['messages'],
	settings?: Partial<
		Omit<openai.CreateChatCompletionRequest, 'messages' | 'model'>
	>,
	customModelName?: string
): Promise<string | undefined> {
	const headers = {
		Authorization: `Bearer ${apiKey}`,
		'Content-Type': 'application/json'
	}
	const actualModel = model === CHAT_MODELS.CUSTOMIZE.name && customModelName 
		? customModelName 
		: (model ?? CHAT_MODELS.GPT_35_TURBO.name)
	const body: openai.CreateChatCompletionRequest = {
		messages,
		model: actualModel,
		...settings
	}
	const requestParam: RequestUrlParam = {
		url: apiUrl,
		method: 'POST',
		contentType: 'application/json',
		body: JSON.stringify(body),
		headers
	}
	console.debug('Calling openAI', requestParam)
	const res: openai.CreateChatCompletionResponse | undefined = await request(
		requestParam
	)
		.then((response) => {
			return JSON.parse(response)
		})
		.catch((err) => {
			console.error(err)
			if (err.code === 429) {
				console.error(
					'OpenAI API rate limit exceeded. If you have free account, your credits may have been consumed or expired.'
				)
			}
		})
	return res?.choices?.[0]?.message?.content
}
