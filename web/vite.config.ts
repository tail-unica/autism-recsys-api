import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const apiTarget = env.VITE_API_PROXY_TARGET || 'http://localhost:8100';
  const backendTarget = env.VITE_BACKEND_PROXY_TARGET || 'http://localhost:3001';

  return {
    plugins: [react()],
    base: '',
    server: {
      host: true,
      port: 3000,
      proxy: {
        '/backend': {
          target: backendTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/backend/, ''),
        },
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  };
});
