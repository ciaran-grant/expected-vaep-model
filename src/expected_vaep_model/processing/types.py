from dataclasses import dataclass, field
from typing import List


@dataclass
class Types:
    action_types: List[str] = field(default_factory=lambda: [
        'Kick', 
        'Handball',
        'Receive',
        'Mark',
        'Gather',
        'Hard Ball Get',
        'Loose Ball Get',
        'Knock On',
        'Spoil',
        'Shot'
    ])
    outcome_types: List[str] = field(default_factory=lambda: [
        'effective',
        'ineffective',
        'clanger',
    ])

