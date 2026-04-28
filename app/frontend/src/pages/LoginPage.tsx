import React, { useState, useCallback } from 'react';

interface LoginPageProps {
    onLoginSuccess: () => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLoginSuccess }) => {
    const [secret, setSecret] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError(null);
        setLoading(true);
        try {
            const response = await fetch('/api/v1/auth/token', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ secret }),
            });
            if (!response.ok) {
                const detail = await response.text();
                throw new Error(detail || 'Invalid secret');
            }
            const data = await response.json();
            localStorage.setItem('klin_jwt', data.access_token);
            onLoginSuccess();
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : 'Login failed';
            setError(message);
        } finally {
            setLoading(false);
        }
    }, [secret, onLoginSuccess]);

    return (
        <div className="login-page">
            <div className="login-card">
                <h2>Авторизация KLIN</h2>
                <form onSubmit={handleSubmit}>
                    <input
                        type="password"
                        placeholder="Введите секретный ключ"
                        value={secret}
                        onChange={e => setSecret(e.target.value)}
                        autoFocus
                        required
                    />
                    <button type="submit" disabled={loading}>
                        {loading ? 'Проверка...' : 'Войти'}
                    </button>
                </form>
                {error && <div className="login-error">{error}</div>}
            </div>
        </div>
    );
};
