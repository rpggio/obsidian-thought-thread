import { App, PluginSettingTab, Setting } from 'obsidian'
import { ChatStreamPlugin } from 'src/ChatStreamPlugin'
import { getModels } from './ChatStreamSettings'
import { CHAT_MODELS } from 'src/openai/chatGPT'

export class SettingsTab extends PluginSettingTab {
	plugin: ChatStreamPlugin

	constructor(app: App, plugin: ChatStreamPlugin) {
		super(app, plugin)
		this.plugin = plugin
	}

	display(): void {
		const { containerEl } = this

		containerEl.empty()

		new Setting(containerEl)
			.setName('Model')
			.setDesc('Select the GPT model to use.')
			.addDropdown((cb) => {
				getModels().forEach((model) => {
					cb.addOption(model, model)
				})
				cb.setValue(this.plugin.settings.apiModel)
				cb.onChange(async (value) => {
					this.plugin.settings.apiModel = value
					await this.plugin.saveSettings()
					// Trigger refresh to show/hide custom model input
					this.display()
				})
			})

		if (this.plugin.settings.apiModel === CHAT_MODELS.CUSTOMIZE.name) {
			new Setting(containerEl)
				.setName('Custom Model Name')
				.setDesc('Enter the name of your custom model')
				.addText((text) => {
					text
						.setPlaceholder('e.g., gpt-4-1106-vision-preview')
						.setValue(this.plugin.settings.customModelName)
						.onChange(async (value) => {
							this.plugin.settings.customModelName = value
							await this.plugin.saveSettings()
						})
				})
		}

		new Setting(containerEl)
			.setName('API key')
			.setDesc('The API key to use when making requests - Get from OpenAI')
			.addText((text) => {
				text.inputEl.type = 'password'
				text
					.setPlaceholder('API Key')
					.setValue(this.plugin.settings.apiKey)
					.onChange(async (value) => {
						this.plugin.settings.apiKey = value
						await this.plugin.saveSettings()
					})
			})

		new Setting(containerEl)
			.setName('System prompt')
			.setDesc(
				`The system prompt sent with each request to the API. \n(Note: you can override this by beginning a note stream with a note starting 'SYSTEM PROMPT'. The remaining content of that note will be used as system prompt.)`
			)
			.addTextArea((component) => {
				component.inputEl.rows = 6
				component.inputEl.style.width = '300px'
				component.inputEl.style.fontSize = '10px'
				component.setValue(this.plugin.settings.systemPrompt)
				component.onChange(async (value) => {
					this.plugin.settings.systemPrompt = value
					await this.plugin.saveSettings()
				})
			})

		new Setting(containerEl)
			.setName('Max input tokens')
			.setDesc(
				'The maximum number of tokens to send (within model limit). 0 means as many as possible'
			)
			.addText((text) =>
				text
					.setValue(this.plugin.settings.maxInputTokens.toString())
					.onChange(async (value) => {
						const parsed = parseInt(value)
						if (!isNaN(parsed)) {
							this.plugin.settings.maxInputTokens = parsed
							await this.plugin.saveSettings()
						}
					})
			)

		new Setting(containerEl)
			.setName('Max response tokens')
			.setDesc(
				'The maximum number of tokens to return from the API. 0 means no limit. (A token is about 4 characters).'
			)
			.addText((text) =>
				text
					.setValue(this.plugin.settings.maxResponseTokens.toString())
					.onChange(async (value) => {
						const parsed = parseInt(value)
						if (!isNaN(parsed)) {
							this.plugin.settings.maxResponseTokens = parsed
							await this.plugin.saveSettings()
						}
					})
			)

		new Setting(containerEl)
			.setName('Max depth')
			.setDesc(
				'The maximum depth of ancestor notes to include. 0 means no limit.'
			)
			.addText((text) =>
				text
					.setValue(this.plugin.settings.maxDepth.toString())
					.onChange(async (value) => {
						const parsed = parseInt(value)
						if (!isNaN(parsed)) {
							this.plugin.settings.maxDepth = parsed
							await this.plugin.saveSettings()
						}
					})
			)

		new Setting(containerEl)
			.setName('Temperature')
			.setDesc('Sampling temperature (0-2). 0 means no randomness.')
			.addText((text) =>
				text
					.setValue(this.plugin.settings.temperature.toString())
					.onChange(async (value) => {
						const parsed = parseFloat(value)
						if (!isNaN(parsed) && parsed >= 0 && parsed <= 2) {
							this.plugin.settings.temperature = parsed
							await this.plugin.saveSettings()
						}
					})
			)

		new Setting(containerEl)
			.setName('API URL')
			.setDesc(
				"The chat completions URL to use. You probably won't need to change this."
			)
			.addText((text) => {
				text.inputEl.style.width = '300px'
				text
					.setPlaceholder('API URL')
					.setValue(this.plugin.settings.apiUrl)
					.onChange(async (value) => {
						this.plugin.settings.apiUrl = value
						await this.plugin.saveSettings()
					})
			})

		new Setting(containerEl)
			.setName('Debug output')
			.setDesc('Enable debug output in the console')
			.addToggle((component) => {
				component
					.setValue(this.plugin.settings.debug)
					.onChange(async (value) => {
						this.plugin.settings.debug = value
						await this.plugin.saveSettings()
					})
			})
	}
}

export default SettingsTab
