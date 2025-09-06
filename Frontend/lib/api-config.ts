const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const endpoints = {
  'upload': '/upload',
  'ask': '/ask',
  'ask_stream': '/ask_stream',
  'test-upload': '/test-upload',
  'summarize': '/summarize',
} as const;

export type EndpointKey = keyof typeof endpoints;

export const API_CONFIG = {
  baseUrl: baseUrl,
  endpoints,
  getUrl: (endpoint: EndpointKey) => `${baseUrl}${endpoints[endpoint]}`,
};
