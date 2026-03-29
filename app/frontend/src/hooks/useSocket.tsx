import { useEffect, useRef, useState } from 'react';

export const useSocket = (url: string) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastFrameData, setLastFrameData] = useState<any>(null);
    const socketRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        const socket = new WebSocket(url);
        socketRef.current = socket;

        socket.onopen = () => setIsConnected(true);
        socket.onclose = () => setIsConnected(false);
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setLastFrameData(data);
        };

        return () => socket.close();
    }, [url]);

    const sendFrame = (frameBase64: string) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({ image: frameBase64 }));
        }
    };

    return { isConnected, lastFrameData, sendFrame };
};