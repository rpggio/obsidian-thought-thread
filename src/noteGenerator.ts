import { TiktokenModel, encodingForModel } from 'js-tiktoken'
import { App, ItemView, Notice } from 'obsidian'
import { CanvasNode } from './obsidian/canvas-internal'
import { CanvasView, calcHeight, createNode } from './obsidian/canvas-patches'
import {
	CHAT_MODELS,
	ChatGPTModel,
	chatModelByName,
	ChatModelSettings,
	getChatGPTCompletion
} from './openai/chatGPT'
import { openai } from './openai/chatGPT-types'
import {
	ChatStreamSettings,
	DEFAULT_SETTINGS
} from './settings/ChatStreamSettings'
import { Logger } from './util/logging'
import { visitNodeAndAncestors } from './obsidian/canvasUtil'
import { readNodeContent } from './obsidian/fileUtil'

/**
 * Color for assistant notes: 6 == purple
 */
const assistantColor = '6'

/**
 * Height to use for placeholder note
 */
const placeholderNoteHeight = 60

/**
 * Height to use for new empty note
 */
const emptyNoteHeight = 100

export function noteGenerator(
	app: App,
	settings: ChatStreamSettings,
	logDebug: Logger
) {
	const canCallAI = () => {
		if (!settings.apiKey) {
			new Notice('Please set your OpenAI API key in the plugin settings')
			return false
		}

		return true
	}

	const nextNote = async () => {
		logDebug('Creating user note')

		const canvas = getActiveCanvas()
		if (!canvas) {
			logDebug('No active canvas')
			return
		}

		await canvas.requestFrame()

		const selection = canvas.selection
		if (selection?.size !== 1) return
		const values = Array.from(selection.values()) as CanvasNode[]
		const node = values[0]

		if (node) {
			const created = createNode(canvas, node, {
				text: '',
				size: { height: emptyNoteHeight }
			})
			canvas.selectOnly(created, true /* startEditing */)

			// startEditing() doesn't work if called immediately
			await canvas.requestSave()
			await sleep(100)

			created.startEditing()
		}
	}

	const getActiveCanvas = () => {
		const maybeCanvasView = app.workspace.getActiveViewOfType(
			ItemView
		) as CanvasView | null
		return maybeCanvasView ? maybeCanvasView['canvas'] : null
	}

	const isSystemPromptNode = (text: string) =>
		text.trim().startsWith('SYSTEM PROMPT')

	const getSystemPrompt = async (node: CanvasNode) => {
		let foundPrompt: string | null = null

		await visitNodeAndAncestors(node, async (n: CanvasNode) => {
			const text = await readNodeContent(n)
			if (text && isSystemPromptNode(text)) {
				foundPrompt = text
				return false
			} else {
				return true
			}
		})

		return foundPrompt || settings.systemPrompt
	}

	const buildMessages = async (node: CanvasNode) => {
		const messages: openai.ChatCompletionRequestMessage[] = []
		const isCustomModel = settings.apiModel === CHAT_MODELS.CUSTOMIZE.name
		const encoding = isCustomModel ? null : getEncoding(settings)
		let tokenCount = 0

		// Note: We are not checking for system prompt longer than context window.
		// That scenario makes no sense, though.
		const systemPrompt = await getSystemPrompt(node)
		if (systemPrompt) {
			if (!isCustomModel && encoding) {
				tokenCount += encoding.encode(systemPrompt).length
			}
			messages.push({
				role: 'system',
				content: systemPrompt
			})
		}

		const visit = async (node: CanvasNode, depth: number) => {
			if (settings.maxDepth && depth > settings.maxDepth) return false

			const nodeData = node.getData()
			let nodeText = (await readNodeContent(node))?.trim() || ''
			const inputLimit = isCustomModel ? Infinity : getTokenLimit(settings)

			let shouldContinue = true
			if (!nodeText) {
				return shouldContinue
			}

			if (nodeText.startsWith('data:image')) {
				messages.unshift({
					content: [{
						'type': 'image_url',
						'image_url': { 'url': nodeText }
					}],
					role: 'user'
				})
			} else {
				if (isSystemPromptNode(nodeText)) return true

				let nodeTokens = isCustomModel || !encoding ? null : encoding.encode(nodeText)
				let keptNodeTokens: number

				if (!isCustomModel && nodeTokens && tokenCount + nodeTokens.length > inputLimit) {
					// will exceed input limit
					shouldContinue = false

					// Leaving one token margin, just in case
					const keepTokens = Math.max(
						0,
						inputLimit - tokenCount - 1
					)
					if (encoding) {
						const keepBytes = encoding
							.decode(nodeTokens.slice(0, keepTokens))
							.length
						nodeText = nodeText.slice(0, keepBytes)
					}
					keptNodeTokens = keepTokens
				} else {
					keptNodeTokens = nodeTokens?.length || 0
				}

				if (!isCustomModel) {
					tokenCount += keptNodeTokens
				}

				messages.unshift({
					content: nodeText,
					role: 'user'
				})
			}

			return shouldContinue
		}

		await visitNodeAndAncestors(node, visit)

		if (messages.length) {
			return { messages, tokenCount }
		} else {
			return { messages: [], tokenCount: 0 }
		}
	}

	const generateNote = async () => {
		if (!canCallAI()) return

		logDebug('Creating AI note')

		const canvas = getActiveCanvas()
		if (!canvas) {
			logDebug('No active canvas')
			return
		}

		await canvas.requestFrame()

		const selection = canvas.selection
		if (selection?.size !== 1) return
		const values = Array.from(selection.values())
		const node = values[0]

		if (node) {
			// Last typed characters might not be applied to note yet
			await canvas.requestSave()
			await sleep(200)

			const { messages, tokenCount } = await buildMessages(node)
			if (!messages.length) return

			const created = createNode(
				canvas,
				node,
				{
					text: `Calling AI (${settings.apiModel})...`,
					size: { height: placeholderNoteHeight }
				},
				{
					color: assistantColor,
					chat_role: 'assistant'
				}
			)

			new Notice(
				`Sending ${messages.length} notes with ${tokenCount} tokens to GPT`
			)

			try {
				logDebug('messages', messages)

				const generated = await getChatGPTCompletion(
					settings.apiKey,
					settings.apiUrl,
					settings.apiModel,
					messages,
					{
						temperature: settings.temperature,
						max_tokens: settings.maxResponseTokens || undefined
					},
					settings.customModelName
				)

				if (generated == null) {
					new Notice(`Empty or unreadable response from GPT`)
					canvas.removeNode(created)
					return
				}

				created.setText(generated)
				const height = calcHeight({
					text: generated,
					parentHeight: node.height
				})
				created.moveAndResize({
					height,
					width: created.width,
					x: created.x,
					y: created.y
				})

				const selectedNoteId =
					canvas.selection?.size === 1
						? Array.from(canvas.selection.values())?.[0]?.id
						: undefined

				if (selectedNoteId === node?.id || selectedNoteId == null) {
					// If the user has not changed selection, select the created node
					canvas.selectOnly(created, false /* startEditing */)
				}
			} catch (error) {
				new Notice(`Error calling GPT: ${error.message || error}`)
				canvas.removeNode(created)
			}

			await canvas.requestSave()
		}
	}

	return { nextNote, generateNote }
}

function getEncoding(settings: ChatStreamSettings) {
	if (settings.apiModel === CHAT_MODELS.CUSTOMIZE.name) {
		return null
	}
	const model: ChatModelSettings | undefined = chatModelByName(settings.apiModel)
	return encodingForModel(
		(model?.encodingFrom || model?.name || DEFAULT_SETTINGS.apiModel) as TiktokenModel
	)
}

function getTokenLimit(settings: ChatStreamSettings) {
	if (settings.apiModel === CHAT_MODELS.CUSTOMIZE.name) {
		return Infinity
	}
	const model = chatModelByName(settings.apiModel) || CHAT_MODELS.GPT_35_TURBO_0125
	return settings.maxInputTokens
		? Math.min(settings.maxInputTokens, model.tokenLimit)
		: model.tokenLimit
}
