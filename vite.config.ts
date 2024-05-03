import { defineConfig } from 'vite'
import Icons from 'unplugin-icons/vite'

export default defineConfig(({ command, mode }) => {
	if (command == 'serve') {
		return {
			base: '/',
		}
	}
  return {
    plugins: [],
		// TODO: Uncomment or unmerged to main.
    base: '/understand-stable-diffusion-slidev-ja/',
  }
})