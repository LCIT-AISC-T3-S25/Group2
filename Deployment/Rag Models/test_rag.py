import requests

BASE_URL = "http://localhost:8004"  # Update if deployed on different host/port


def send_question_and_validate(question):
    response = requests.post(f"{BASE_URL}/predict", json={"text": question})
    assert response.status_code == 200, f"❌ Failed on: {question}"
    data = response.json()
    assert "answer" in data, f"❌ No 'answer' in response for: {question}"
    print(f"✅ Passed: {question}")


def test_valid_questions():
    questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension diagnosed?",
        "What is the treatment for asthma?",
        "Can COVID-19 cause long-term effects?",
        "What are the side effects of paracetamol?",
        "Is there a vaccine for malaria?",
        "What is the difference between a virus and a bacterium?",
        "How does insulin regulate blood sugar?",
        "What is the function of the kidney?",
        "Explain the role of white blood cells.",
        "How do antibiotics work?",
        "What causes anemia?",
        "What is gene therapy?",
        "What are the stages of cancer?",
        "What is the recommended dosage of ibuprofen for adults?",
        "Can high cholesterol lead to heart disease?",
        "What is the use of MRI in diagnosis?",
        "Explain the process of wound healing.",
        "What is Alzheimer's disease?",
        "What are common symptoms of depression?",
        "What is the function of the liver?",
        "What does this Python code do: for i in range(10): print(i)?"
    ]

    for q in questions:
        send_question_and_validate(q)


def test_empty_question():
    response = requests.post(f"{BASE_URL}/predict", json={"text": ""})
    assert response.status_code in [400, 422]
    print("✅ test_empty_question passed.")


def test_non_json_input():
    response = requests.post(f"{BASE_URL}/predict", data="This is not JSON")
    assert response.status_code in [400, 422]
    print("✅ test_non_json_input passed.")


def test_unanswerable_question():
    response = requests.post(f"{BASE_URL}/predict", json={"text": "What is the capital of Mars?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    print("✅ test_unanswerable_question passed.")


def test_large_input():
    long_text = "Explain the human genome project. " * 100
    response = requests.post(f"{BASE_URL}/predict", json={"text": long_text})
    assert response.status_code == 200
    assert "answer" in response.json()
    print("✅ test_large_input passed.")


def test_sql_injection_attempt():
    response = requests.post(f"{BASE_URL}/predict", json={"text": "SELECT * FROM users;"})
    assert response.status_code == 200
    assert "answer" in response.json()
    print("✅ test_sql_injection_attempt passed.")


if __name__ == "__main__":
    test_valid_questions()
    test_empty_question()
    test_non_json_input()
    test_unanswerable_question()
    test_large_input()
    test_sql_injection_attempt()
