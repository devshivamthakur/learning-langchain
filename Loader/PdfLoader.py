from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Load the pdf file
loader = PyPDFLoader("Loader/Shivam_Kumar_Thakur_resume.pdf")

# Load the documents
docs = loader.load()

# Combine all pages into a single text for better context
full_text = "\n".join([doc.page_content for doc in docs])

# Define the pydantic model for structured output
class ResumeInfo(BaseModel):
    name: str = Field(..., description="Full name of the candidate")
    email: str = Field(..., description="Email address of the candidate")
    phone_number: str = Field(..., description="Phone number of the candidate")
    skills: list[str] = Field(..., description="List of skills")
    education: list[str] = Field(..., description="List of educational qualifications")
    experience: list[str] = Field(..., description="List of work experiences")
    projects: list[str] = Field(..., description="List of projects")
    certifications: list[str] = Field(..., description="List of certifications")
    achievements: list[str] = Field(..., description="List of achievements")
    hobbies: list[str] = Field(..., description="List of hobbies")
    languages_known: list[str] = Field(..., description="List of languages known")
    linkedin_profile: str = Field(..., description="LinkedIn profile URL")
    github_profile: str = Field(..., description="GitHub profile URL")
    summary: str = Field(..., description="Professional summary or objective")
    address: str = Field(..., description="Residential address")

# Initialize the output parser
output_parser = PydanticOutputParser(pydantic_object=ResumeInfo)

# Improved prompt template
prompt = PromptTemplate(
    input_variables=["text"],
    template="""Extract the following information from the resume text provided below. 
If any information is not available, use "Not provided" as the value.

RESUME TEXT:
{text}

EXTRACTION INSTRUCTIONS:
- Extract the full name from the header or top section of the resume
- For lists (skills, education, etc.), extract all items you can find
- Be precise and extract only the information present in the text
- If you cannot find certain information, use "Not provided"

{format_instructions}

Extracted Information:""",
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

# Initialize the HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Next-80B-A3B-Instruct',
    task='text-generation',
    provider="auto",
    max_new_tokens=512,  # Increased for better extraction
    temperature=0.1,     # Lower temperature for more consistent results
    top_p=0.9,
)

model = ChatHuggingFace(llm=llm)

resumeInfo = None

# Process the combined text
print("Processing resume...")
try:
    chain = prompt | model | output_parser
    result = chain.invoke({"text": full_text})
    resumeInfo = result
except Exception as e:
    print(f"Error during extraction: {e}")
    # Fallback: try with individual pages
    print("Trying with individual pages...")
    for i, doc in enumerate(docs):
        print(f"Processing page {i+1}...")
        try:
            chain = prompt | model | output_parser
            result = chain.invoke({"text": doc.page_content})
            if result.name and result.name != "Not provided":
                print(f"Found name on page {i+1}: {result.name}")
                break
        except Exception as e:
            print(f"Error processing page {i+1}: {e}")


print("Extracted Resume Information:")
print(resumeInfo)