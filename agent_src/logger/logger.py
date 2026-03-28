
class BaseLogger(ABC):
    def log_event(self, event_type: str, payload: dict[str, Any]) -> None: ...

class Logger(BaseLogger):
    """Simple logger that prints events to the console."""

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        print(f"[{event_type}] {json.dumps(payload, ensure_ascii=False)}")

class jsonLogger(BaseLogger):
    """Logger that writes events to a JSONL file."""

    def __init__(self, output_path: Path, message_manager: MessageManager) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            json.dump({"event_type": event_type, "payload": payload}, f, ensure_ascii=False)
            f.write("\n")