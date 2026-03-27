# Clean Architecture Isolation View

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "curve": "basis",
    "nodeSpacing": 40,
    "rankSpacing": 48
  },
  "themeVariables": {
    "fontFamily": "Georgia, Times New Roman, serif",
    "fontSize": "17px",
    "lineColor": "#5b6572",
    "primaryTextColor": "#17202a",
    "secondaryTextColor": "#17202a",
    "tertiaryTextColor": "#17202a",
    "background": "#fffdf8"
  }
}}%%
flowchart LR
    subgraph view[" "]
        direction TB

        subgraph ring4["Ring 4  Infrastructure and Frameworks"]
            direction TB
            infra["app.infrastructure.*<br/>repositories / processors / storage / producers"]
            config["app.config<br/>settings / env resolution"]
            external["Postgres / RabbitMQ / S3 / Triton / OpenCV / SQLAlchemy"]

            subgraph ring3["Ring 3  Interface Adapters"]
                direction TB
                adapters["app.presentation.* and app.ioc<br/>controllers / queue workers / composition root"]

                subgraph ring2["Ring 2  Application"]
                    direction TB
                    application["app.application.*<br/>services / ports / DTOs / mappers / StreamEventConsumer"]

                    subgraph ring1["Ring 1  Kernel-ish Core"]
                        direction TB
                        kernel["app.models and app.application.exceptions<br/>KlinModel / KlinStreamState / result models / ProcessingState / errors"]
                    end
                end
            end
        end
    end

    subgraph gaps["Current isolation gaps"]
        direction TB
        gap1["presentation -> config<br/>direct settings read"]
        gap2["mappers -> models<br/>ORM shaped read mapping"]
        gap3["consumer -> models<br/>persistence model creation"]
        gap4["models -> SQLAlchemy semantics<br/>core is not persistence free"]
    end

    external --> infra
    config --> infra
    infra --> adapters
    adapters --> application
    application --> kernel

    gap1 -.-> adapters
    gap1 -.-> config
    gap2 -.-> application
    gap2 -.-> kernel
    gap3 -.-> application
    gap3 -.-> kernel
    gap4 -.-> kernel
    gap4 -.-> external

    classDef outerLayer fill:#f6ece4,stroke:#8b6b53,stroke-width:3px,color:#2f241d;
    classDef adapterLayer fill:#e9f1fb,stroke:#5c7ea5,stroke-width:3px,color:#1d2b3d;
    classDef appLayer fill:#e9f5ee,stroke:#5d8a72,stroke-width:3px,color:#1d3025;
    classDef kernelLayer fill:#f8f1d9,stroke:#a78538,stroke-width:3px,color:#35280e;
    classDef leak fill:#fff4ef,stroke:#b85f45,stroke-width:2px,color:#48261d;

    class infra,config,external outerLayer;
    class adapters adapterLayer;
    class application appLayer;
    class kernel kernelLayer;
    class gap1,gap2,gap3,gap4 leak;

    style view fill:#fffdf8,stroke:#ffffff,stroke-width:0px
    style ring1 fill:#fff7de,stroke:#c59a3c,stroke-width:4px,rx:40,ry:40,color:#35280e
    style ring2 fill:#f3fbf6,stroke:#6fa184,stroke-width:4px,rx:48,ry:48,color:#1d3025
    style ring3 fill:#f4f8fd,stroke:#7a97ba,stroke-width:4px,rx:56,ry:56,color:#1d2b3d
    style ring4 fill:#fdf8f2,stroke:#9b7d63,stroke-width:4px,rx:64,ry:64,color:#2f241d
    style gaps fill:#fffaf6,stroke:#d3b5a6,stroke-width:2px,rx:24,ry:24,color:#48261d

    linkStyle 5,6,7,8,9,10,11,12 stroke:#c76849,stroke-width:2px,stroke-dasharray: 7 5
```

## Reading This Diagram

- Solid arrows show the intended inward dependency direction: outer layers depend on inner layers.
- Dashed arrows show the places where the current code still leaks persistence or framework concerns across layers.
- This view is intentionally compressed into one block per ring. Use the other flowcharts for DTO-level and processor-level detail.

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
