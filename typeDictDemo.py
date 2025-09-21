from typing import TypedDict

class student(TypedDict):
    name: str
    age: int


devil: student = {
    'age': 233,
    'name': 2000
}    

print(devil)