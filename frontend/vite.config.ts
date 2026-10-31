import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const proxy = {
  '/api': 'http://127.0.0.1:8000',
  '/health': 'http://127.0.0.1:8000',
  '/ws': { target: 'http://127.0.0.1:8000', ws: true },
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    strictPort: true,
    host: true,
    proxy,
  },
  preview: { port: 3000, strictPort: true, proxy },
});
