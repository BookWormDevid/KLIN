import React, { createContext, useContext } from 'react';
import { HttpClient } from '../infrastructure/http/httpClient';
import { ApiVideoRepository } from '../infrastructure/repositories/apiVideoRepository';
import { ApiStreamRepository } from '../infrastructure/repositories/apiStreamRepository';
import { ApiFeedbackRepository } from '../infrastructure/repositories/apiFeedbackRepository';
import { LocalStorageHistoryRepository } from '../infrastructure/repositories/localStorageHistoryRepository';

export interface Services {
    videoRepo: ApiVideoRepository;
    streamRepo: ApiStreamRepository;
    feedbackRepo: ApiFeedbackRepository;
    historyRepo: LocalStorageHistoryRepository;
}

const defaultHttp = new HttpClient();

export const defaultServices: Services = {
    videoRepo: new ApiVideoRepository(defaultHttp),
    streamRepo: new ApiStreamRepository(defaultHttp),
    feedbackRepo: new ApiFeedbackRepository(),   // без аргумента
    historyRepo: new LocalStorageHistoryRepository(),
};

const ServicesContext = createContext<Services>(defaultServices);

export const useServices = () => useContext(ServicesContext);

export const ServicesProvider: React.FC<{ services: Services; children: React.ReactNode }> = ({ services, children }) => (
    <ServicesContext.Provider value={services}>
        {children}
    </ServicesContext.Provider>
);
