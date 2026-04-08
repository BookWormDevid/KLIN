import { appConfig } from '../../../config/appConfig';

export class HttpClient {
    private baseUrl: string;
    private timeout: number;

    constructor(baseUrl?: string, timeout?: number) {
        this.baseUrl = baseUrl || appConfig.apiBaseUrl;
        this.timeout = timeout || appConfig.requestTimeoutMs;
    }

    private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
            });
            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorText = await response.text();
                throw new Error(errorText || `HTTP ${response.status}`);
            }
            return await response.json();
        } catch (err) {
            if (err.name === 'AbortError') {
                throw new Error(`Request timeout (${this.timeout}ms)`);
            }
            throw err;
        }
    }

    get<T>(endpoint: string): Promise<T> {
        return this.request<T>(endpoint, { method: 'GET' });
    }

    post<T>(endpoint: string, body?: any): Promise<T> {
        return this.request<T>(endpoint, {
            method: 'POST',
            body: body ? JSON.stringify(body) : undefined,
        });
    }

    postForm<T>(endpoint: string, formData: FormData): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        return fetch(url, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        }).then(async (response) => {
            clearTimeout(timeoutId);
            if (!response.ok) {
                let errorText = await response.text();
                throw new Error(errorText || `HTTP ${response.status}`);
            }
            return response.json();
        });
    }
}