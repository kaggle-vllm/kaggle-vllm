"""Research evidence errors."""


class ResearchEvidenceError(ValueError):
    """Required research evidence is absent, malformed, or inconsistent."""


class CommunicationFitError(ResearchEvidenceError):
    """Measured observations cannot support the requested communication fit."""


class ProvenanceError(ResearchEvidenceError):
    """A provenance record or content digest is invalid."""
