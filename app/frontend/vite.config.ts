import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

function startupMessage() {
  return {
    name: 'startup-message',
    configureServer(server: any) {
      server.httpServer?.once('listening', () => {
        console.log('\n Основная страница:  http://localhost:5173/')
        console.log(' Тестовая страница: http://localhost:5173/mock.html\n')
      })
    }
  }
}

export default defineConfig({
  plugins: [react(), startupMessage()],
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
      },
    },
  },
})
