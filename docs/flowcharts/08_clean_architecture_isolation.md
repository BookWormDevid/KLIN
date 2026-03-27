# Clean Architecture Isolation View

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontFamily": "Georgia, Times New Roman, serif",
    "fontSize": "16px",
    "lineColor": "#6c6f73",
    "primaryTextColor": "#1f2328",
    "secondaryTextColor": "#1f2328",
    "tertiaryTextColor": "#1f2328",
    "background": "#fffdf8"
  }
}}%%
flowchart TB
    subgraph legend["Reading the map"]
        l1["solid line<br/>main dependency direction"]
        l2["dashed line<br/>architectural leak / coupling"]
    end

    subgraph R4["Outer Ring - Infrastructure / Frameworks / Drivers"]
        infra["app.infrastructure.*<br/>database / repository<br/>producers / storage / processors"]
        config["app.config<br/>Settings / env resolution"]
        externals["External systems & frameworks<br/>Postgres / RabbitMQ / S3 / Triton<br/>aiohttp / boto3 / cv2 / SQLAlchemy"]
    end

    subgraph R3["Ring 3 - Interface Adapters / Entry Points"]
        presentation["app.presentation.*<br/>Litestar controllers<br/>FastStream workers"]
        ioc["app.ioc<br/>composition root / DI wiring"]
    end

    subgraph R2["Ring 2 - Application / Use Cases / Ports"]
        services["app.application.services<br/>KlinService / StreamService"]
        ports["app.application.interfaces<br/>IKlinTaskRepository / IStreamStateRepository / IStreamEventRepository<br/>IKlinRuntimeSettings / IKlinInference / IKlinStream / ..."]
        dto["app.application.dto<br/>request / queue / response contracts"]
        mappers["app.application.mappers<br/>to_klin_read_dto / to_stream_read_dto"]
        consumers["app.application.consumers<br/>StreamEventConsumer"]
    end

    subgraph R1["Inner Ring - Kernel-ish Core"]
        models["app.models<br/>KlinModel / KlinStreamState<br/>KlinYoloResult / KlinMaeResult / KlinX3DResult<br/>ProcessingState"]
        exceptions["app.application.exceptions<br/>KlinNotFoundError / KlinEnqueueError"]
    end

    presentation --> services
    presentation --> dto
    presentation --> ports
    presentation --> mappers

    ioc --> presentation
    ioc --> services
    ioc --> ports
    ioc --> infra
    ioc --> config

    infra --> ports
    infra --> dto
    infra --> models
    infra --> config
    externals --> infra
    config --> ports

    services --> ports
    services --> dto
    services --> models
    services --> exceptions
    services --> mappers

    mappers --> dto
    mappers --> models

    consumers --> ports
    consumers --> dto
    consumers --> models

    models -. orm coupling .-> externals
    consumers -. persistence model creation .-> models
    mappers -. orm based mapping .-> models
    presentation -. direct config read .-> config

    classDef ring1 fill:#f7f1e1,stroke:#8b6f3d,stroke-width:2px,color:#2d2418;
    classDef ring2 fill:#e7f0ec,stroke:#4f7a68,stroke-width:2px,color:#1f312b;
    classDef ring3 fill:#e9eef6,stroke:#58708f,stroke-width:2px,color:#1f2c3c;
    classDef ring4 fill:#f3e8e8,stroke:#8c5b5b,stroke-width:2px,color:#3b2424;
    classDef legendNode fill:#fff8d9,stroke:#a88b2f,stroke-width:1.5px,color:#3b3210;

    class models,exceptions ring1;
    class services,ports,dto,mappers,consumers ring2;
    class presentation,ioc ring3;
    class infra,config,externals ring4;
    class l1,l2 legendNode;

    style R1 fill:#fff8e8,stroke:#b89656,stroke-width:3px,color:#2d2418
    style R2 fill:#f4fbf7,stroke:#6f9c88,stroke-width:3px,color:#1f312b
    style R3 fill:#f5f8fc,stroke:#7b94b4,stroke-width:3px,color:#1f2c3c
    style R4 fill:#fdf5f5,stroke:#b07a7a,stroke-width:3px,color:#3b2424
    style legend fill:#fffdf5,stroke:#d6c27a,stroke-width:2px,color:#3b3210

    linkStyle 25,26,27,28 stroke:#b86b4b,stroke-width:2px,stroke-dasharray: 7 5
```

## Reading This Diagram

- Solid arrows show the current main dependency direction in the project.
- Dashed arrows highlight places where the architecture is still not fully isolated from persistence or framework concerns.

## Isolation Verdict

### Stronger parts

- `KlinService` no longer imports `app_settings` directly; retry settings now come through `IKlinRuntimeSettings`.
- `app.application.dto` is cleaner: read DTOs no longer contain `from_model()` / `from_stream_state()` methods.
- Repository responsibilities are clearer after splitting the active ports into `IKlinTaskRepository`, `IStreamStateRepository`, and `IStreamEventRepository`.
- `app.ioc` still acts as a clean composition root at the edge.

### Weaker parts

- `app.models` are still SQLAlchemy ORM entities, so the innermost layer is not a pure domain kernel.
- `app.application.mappers` still reads ORM models directly, so the read side is cleaner than before, but not persistence-independent.
- `StreamEventConsumer` still constructs persistence-oriented models like `KlinMaeResult` and `KlinYoloResult` directly.
- `KlinController._build_object_key()` still reads `app_settings` directly in the presentation layer.

## Short Conclusion

The project is now closer to a disciplined ports-and-adapters structure than before, but it is still not a strict clean architecture.

The most useful next steps from here would be:

- replace ORM entities in inner layers with pure domain entities
- introduce typed stream event payloads instead of `type: str` + `payload: dict`
- move controller-side config reads behind small injected ports
- stop constructing persistence models directly inside the application consumer
