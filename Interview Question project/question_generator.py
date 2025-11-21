from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load small model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Dropdown-style options
job_roles = [
    "Python Developer",
    "Data Analyst",
    "Machine Learning Engineer",
    "Web Developer",
    "DevOps Engineer",
    "UI/UX Designer",
    "Cybersecurity Analyst",
    "Business Analyst",
    "HR Manager",
    "Project Manager"
]

# Show menu
print("📋 Select a Job Role:")
for i, role in enumerate(job_roles, start=1):
    print(f"{i}. {role}")

# Get choice
while True:
    try:
        choice = int(input("\nEnter your choice (1–10): "))
        if 1 <= choice <= 10:
            job_role = job_roles[choice - 1]
            break
        else:
            print("Please enter a number between 1 and 10.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Prompt for generation
prompt = (
    f"You are an AI interviewer. Generate 10 unique technical interview questions "
    f"for the role of a {job_role}.\n\n1."
)

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# Decode and clean output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
lines = generated_text.split('\n')

questions = []
for line in lines:
    line = line.strip()
    if '?' in line and len(line.split()) > 3:
        cleaned = line.strip("•-1234567890. ").strip()
        if cleaned not in questions:
            questions.append(cleaned)

# Display final output
print(f"\n📌 Interview Questions for {job_role}:\n")
for q in questions[:10]:
    print("•", q)
