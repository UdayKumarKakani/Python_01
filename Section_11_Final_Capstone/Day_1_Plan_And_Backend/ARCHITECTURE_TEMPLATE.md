# Capstone Architecture — <Project Name>

## 1. Problem
<What painful thing does this solve? 2 sentences.>

## 2. Users
<Who uses it? How many do you expect at demo time?>

## 3. User flow
1. ...
2. ...
3. ...

## 4. Architecture diagram
![Architecture](./architecture.png)

## 5. Stack
| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI | ... |
| Auth | JWT (from Section 2) | ... |
| Vector DB | ... | ... |
| LLM | ... | ... |
| Observability | Langfuse | ... |
| Deploy | Render | ... |

## 6. Data
- Source: ...
- Volume: ...
- Ingestion: batch / incremental / streaming
- License / privacy: ...

## 7. Cost model
- DAU: ...
- Queries/user/day: ...
- Cost/day: $...
- Cost/month: $...
- Cost per active user per month: $...

## 8. Risks & unknowns
- ...
- ...
- ...

## 9. Milestones
- **Day 2**: backend + auth + healthz + CI green
- **Day 3**: RAG pipeline works locally on real docs
- **Day 4**: agent workflow works locally end-to-end
- **Day 5**: deployed + observed
- **Day 6**: demo recorded + cost analysis + portfolio page
