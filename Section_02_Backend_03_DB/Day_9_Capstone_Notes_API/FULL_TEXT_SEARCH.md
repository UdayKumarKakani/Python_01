
### 1. Router — `routers/notes.py`

Add `q` to `list_notes` and filter with `Note.title.ilike` / `Note.content.ilike`.

```python
from sqlalchemy import or_, select

@router.get("/", response_model=List[NoteOut])
def list_notes(
    q: str | None = Query(None, min_length=1, max_length=200,
                          description="Search across title and content"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[Note]:
    """List notes belonging to the current user, paginated. Optional text search."""
    stmt = select(Note).where(Note.owner_id == current_user.id)

    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(or_(Note.title.ilike(pattern),
                              Note.content.ilike(pattern)))

    stmt = stmt.order_by(Note.id.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())
```

### 2. Try it

```bash
curl -s "http://127.0.0.1:8765/v1/notes/?q=grocery" \
     -H "Authorization: Bearer $TOK"
```




## Optional: also match tag names

If you want `?q=urgent` to match notes tagged `urgent`, add to Approach A's
filter:

```python
from models import Tag

stmt = stmt.outerjoin(Note.tags).where(
    or_(Note.title.ilike(pattern),
        Note.content.ilike(pattern),
        Tag.name.ilike(pattern))
).distinct()
```
