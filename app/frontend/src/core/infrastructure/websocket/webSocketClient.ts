export class WebSocketClient {
    private socket: WebSocket | null = null;
    private url: string;
    private messageHandlers: ((data: string) => void)[] = [];

    constructor(url: string) {
        this.url = url;
    }

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.socket = new WebSocket(this.url);
            this.socket.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            this.socket.onerror = (err) => {
                console.error('WebSocket error', err);
                reject(err);
            };
            this.socket.onmessage = (event) => {
                this.messageHandlers.forEach(handler => handler(event.data));
            };
        });
    }

    disconnect(): void {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }

    send(data: string): void {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(data);
        } else {
            console.warn('WebSocket not open, cannot send');
        }
    }

    onMessage(handler: (data: string) => void): () => void {
        this.messageHandlers.push(handler);
        return () => {
            this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
        };
    }

    isConnected(): boolean {
        return this.socket !== null && this.socket.readyState === WebSocket.OPEN;
    }
}