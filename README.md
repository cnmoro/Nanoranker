*   pip install nanoranker

```python
from nanoranker import rank

query = "Who directed 'Inception'?"
documents = [
    "'Inception' is a 2010 science fiction film directed by Christopher Nolan. It explores the concept of dream invasion and manipulation.",
    "Steven Spielberg is one of the most well-known directors of all time, famous for films like 'E.T.', 'Jaws', and 'Jurassic Park'.",
    "'Titanic', directed by James Cameron, was released in 1997 and became one of the highest-grossing films of all time.",
    "Christopher Nolan is a British-American filmmaker known for his cerebral and nonlinear storytelling in movies like 'Memento', 'The Dark Knight', and 'Inception'.",
    "Martin Scorsese directed the crime drama 'Goodfellas', which is considered a masterpiece in the gangster film genre."
]

rank(query, documents)
# Output:
# [("'Inception' is a 2010 science fiction film directed by Christopher Nolan. It explores the concept of dream invasion and manipulation.",
#   0.30231907351694276),
#  ("'Titanic', directed by James Cameron, was released in 1997 and became one of the highest-grossing films of all time.",
#   0.19274971230156498),
#  ("Steven Spielberg is one of the most well-known directors of all time, famous for films like 'E.T.', 'Jaws', and 'Jurassic Park'.",
#   0.19266481513294043),
#  ("Martin Scorsese directed the crime drama 'Goodfellas', which is considered a masterpiece in the gangster film genre.",
#   0.1603303513065687),
#  ("Christopher Nolan is a British-American filmmaker known for his cerebral and nonlinear storytelling in movies like 'Memento', 'The Dark Knight', and 'Inception'.",
#   0.15193604774198313)]

query = "What is the speed of light?"
documents = [
    "The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s), or about 186,282 miles per second.",
    "Isaac Newton's laws of motion and gravity laid the groundwork for classical mechanics.",
    "The theory of relativity, proposed by Albert Einstein, has revolutionized our understanding of space, time, and gravity.",
    "The Earth orbits the Sun at an average distance of about 93 million miles, taking roughly 365.25 days to complete one revolution.",
    "Light can be described as both a wave and a particle, a concept known as wave-particle duality."
]
# Output:
# [("Isaac Newton's laws of motion and gravity laid the groundwork for classical mechanics.",
#   0.26837394568994555),
#  ('The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s), or about 186,282 miles per second.',
#   0.22389275760593016),
#  ('Light can be described as both a wave and a particle, a concept known as wave-particle duality.',
#   0.2190827494212158),
#  ('The Earth orbits the Sun at an average distance of about 93 million miles, taking roughly 365.25 days to complete one revolution.',
#   0.15447457049559693),
#  ('The theory of relativity, proposed by Albert Einstein, has revolutionized our understanding of space, time, and gravity.',
#   0.13417597678731155)]

query = "Who wrote 'Pride and Prejudice'?"
documents = [
    "Pride and Prejudice is a novel written by Jane Austen, first published in 1813. It is a classic of English literature.",
    "Charlotte Brontë, known for her novel Jane Eyre, was a 19th-century English novelist.",
    "William Shakespeare is often considered the greatest playwright in the English language, famous for works such as Hamlet, Romeo and Juliet, and Macbeth.",
    "Pride and Prejudice explores themes of love, social status, and individual growth, set in the British Regency era.",
    "Jane Austen, an English novelist, is renowned for her works that critique the British landed gentry of the 18th century."
]
# Output:
# [('Pride and Prejudice explores themes of love, social status, and individual growth, set in the British Regency era.',
#   0.29640036907108874),
#  ('William Shakespeare is often considered the greatest playwright in the English language, famous for works such as Hamlet, Romeo and Juliet, and Macbeth.',
#   0.22162575844903826),
#  ('Jane Austen, an English novelist, is renowned for her works that critique the British landed gentry of the 18th century.',
#   0.21914652110923447),
#  ('Pride and Prejudice is a novel written by Jane Austen, first published in 1813. It is a classic of English literature.',
#   0.143248682405217),
#  ('Charlotte Brontë, known for her novel Jane Eyre, was a 19th-century English novelist.',
#   0.11957866896542155)]
```