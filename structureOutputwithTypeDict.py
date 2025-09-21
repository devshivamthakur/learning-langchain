from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, TypedDict
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
load_dotenv()

llm  = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,

)

# llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', temperature=0.7)
llm = ChatHuggingFace(llm = llm)
json_schema = {
    "title": "StudentsList",
    "type": "object",
    "properties": {
        "students": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "student_id": {
                        "type": "string",
                        "description": "Unique identifier for the student"
                    },
                    "first_name": {
                        "type": "string",
                        "description": "First name of the student"
                    },
                    "last_name": {
                        "type": "string",
                        "description": "Last name of the student"
                    },
                    "date_of_birth": {
                        "type": "string",
                        "format": "date",
                        "description": "Date of birth in YYYY-MM-DD format"
                    },
                    "email": {
                        "type": "string",
                        "format": "email",
                        "description": "Student's email address"
                    },
                    "phone": {
                        "type": "string",
                        "description": "Student's phone number"
                    },
                    "address": {
                        "type": "object",
                        "properties": {
                            "street": { "type": "string" },
                            "city": { "type": "string" },
                            "state": { "type": "string" },
                            "postal_code": { "type": "string" },
                            "country": { "type": "string" }
                        },
                        "required": ["street", "city", "state", "postal_code", "country"]
                    },
                    "enrollment_year": {
                        "type": "integer",
                        "description": "Year the student enrolled"
                    },
                    "major": {
                        "type": "string",
                        "description": "Student's major or field of study"
                    },
                    "courses": {
                        "type": "array",
                        "description": "List of courses the student is enrolled in",
                        "items": {
                            "type": "object",
                            "properties": {
                                "course_id": { "type": "string" },
                                "course_name": { "type": "string" },
                                "grade": { "type": ["string", "null"] }
                            },
                            "required": ["course_id", "course_name"]
                        }
                    },
                    "is_active": {
                        "type": "boolean",
                        "description": "Whether the student is currently active"
                    }
                },
                "required": [
                    "student_id",
                    "first_name",
                    "last_name",
                    "date_of_birth",
                    "email",
                    "enrollment_year",
                    "major",
                    "is_active"
                ]
            }
        }
    },
    "required": ["students"]
}

structure_output = llm.with_structured_output(json_schema)
result = structure_output.invoke("generate 5 student dummy data having information")
# Assuming `result` is the output from structure_output.invoke(...)
students = result["students"]

for student in students:
    print("Student ID:", student.get("student_id"))
    print("Name:", student.get("first_name"), student.get("last_name"))
    print("DOB:", student.get("date_of_birth"))
    print("Email:", student.get("email"))
    print("Phone:", student.get("phone"))
    
    address = student.get("address", {})
    print("Address:", f"{address.get('street')}, {address.get('city')}, {address.get('state')}, {address.get('postal_code')}, {address.get('country')}")

    print("Enrolled Year:", student.get("enrollment_year"))
    print("Major:", student.get("major"))
    print("Is Active:", student.get("is_active"))

    print("Courses:")
    for course in student.get("courses", []):
        print(f"  - {course.get('course_name')} ({course.get('course_id')}) - Grade: {course.get('grade')}")
    
    print("-" * 40)
