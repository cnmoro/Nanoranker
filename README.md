[![Downloads](https://static.pepy.tech/badge/nanoranker)](https://pepy.tech/project/nanoranker)

[![Downloads](https://static.pepy.tech/badge/nanoranker/month)](https://pepy.tech/project/nanoranker)

[![Downloads](https://static.pepy.tech/badge/nanoranker/week)](https://pepy.tech/project/nanoranker)


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
#   0.3140751875733907),
#  ("Christopher Nolan is a British-American filmmaker known for his cerebral and nonlinear storytelling in movies like 'Memento', 'The Dark Knight', and 'Inception'.",
#   0.2776111024668988),
#  ("'Titanic', directed by James Cameron, was released in 1997 and became one of the highest-grossing films of all time.",
#   0.23555834379639545),
#  ("Martin Scorsese directed the crime drama 'Goodfellas', which is considered a masterpiece in the gangster film genre.",
#   0.13833970116570868),
#  ("Steven Spielberg is one of the most well-known directors of all time, famous for films like 'E.T.', 'Jaws', and 'Jurassic Park'.",
#   0.03441566499760637)]

query = "What is the speed of light?"
documents = [
    "The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s), or about 186,282 miles per second.",
    "Isaac Newton's laws of motion and gravity laid the groundwork for classical mechanics.",
    "The theory of relativity, proposed by Albert Einstein, has revolutionized our understanding of space, time, and gravity.",
    "The Earth orbits the Sun at an average distance of about 93 million miles, taking roughly 365.25 days to complete one revolution.",
    "Light can be described as both a wave and a particle, a concept known as wave-particle duality."
]
rank(query, documents)
# Output:
# [('The speed of light in a vacuum is approximately 299,792 kilometers per second (km/s), or about 186,282 miles per second.',
#   0.30580432492241644),
#  ('Light can be described as both a wave and a particle, a concept known as wave-particle duality.',
#   0.28595544542990103),
#  ('The Earth orbits the Sun at an average distance of about 93 million miles, taking roughly 365.25 days to complete one revolution.',
#   0.26599248433979883),
#  ("Isaac Newton's laws of motion and gravity laid the groundwork for classical mechanics.",
#   0.07144421109976119),
#  ('The theory of relativity, proposed by Albert Einstein, has revolutionized our understanding of space, time, and gravity.',
#   0.07080353420812247)]

query = "Who wrote 'Pride and Prejudice'?"
documents = [
    "Pride and Prejudice is a novel written by Jane Austen, first published in 1813. It is a classic of English literature.",
    "Charlotte Brontë, known for her novel Jane Eyre, was a 19th-century English novelist.",
    "William Shakespeare is often considered the greatest playwright in the English language, famous for works such as Hamlet, Romeo and Juliet, and Macbeth.",
    "Pride and Prejudice explores themes of love, social status, and individual growth, set in the British Regency era.",
    "Jane Austen, an English novelist, is renowned for her works that critique the British landed gentry of the 18th century."
]
rank(query, documents)
# Output:
# [('Pride and Prejudice is a novel written by Jane Austen, first published in 1813. It is a classic of English literature.',
#   0.3413168531560433),
#  ('Pride and Prejudice explores themes of love, social status, and individual growth, set in the British Regency era.',
#   0.31806861613354287),
#  ('Jane Austen, an English novelist, is renowned for her works that critique the British landed gentry of the 18th century.',
#   0.19975825998268765),
#  ('William Shakespeare is often considered the greatest playwright in the English language, famous for works such as Hamlet, Romeo and Juliet, and Macbeth.',
#   0.07942214548601552),
#  ('Charlotte Brontë, known for her novel Jane Eyre, was a 19th-century English novelist.',
#   0.06143412524171063)]

query = "Quem escreveu 'Dom Casmurro'?"
documents = [
    "'Dom Casmurro' é um romance escrito por Machado de Assis, publicado pela primeira vez em 1899. É considerado uma das obras-primas da literatura brasileira.",
    "Machado de Assis, um dos maiores escritores da literatura brasileira, é autor de obras como 'Dom Casmurro', 'Memórias Póstumas de Brás Cubas' e 'Quincas Borba'.",
    "'O Guarani', um romance escrito por José de Alencar, é um marco do romantismo no Brasil e foi publicado em 1857.",
    "Clarice Lispector foi uma importante escritora brasileira, conhecida por obras como 'A Hora da Estrela' e 'Perto do Coração Selvagem'.",
    "'Dom Casmurro' narra a vida de Bento Santiago, conhecido como Bentinho, e seus complexos sentimentos de amor e ciúmes em relação a Capitu."
]
rank(query, documents)
# Output:
# [("'Dom Casmurro' é um romance escrito por Machado de Assis, publicado pela primeira vez em 1899. É considerado uma das obras-primas da literatura brasileira.",
#   0.3487686026605069),
#  ("'Dom Casmurro' narra a vida de Bento Santiago, conhecido como Bentinho, e seus complexos sentimentos de amor e ciúmes em relação a Capitu.",
#   0.24520132359838245),
#  ("Machado de Assis, um dos maiores escritores da literatura brasileira, é autor de obras como 'Dom Casmurro', 'Memórias Póstumas de Brás Cubas' e 'Quincas Borba'.",
#   0.2189232997713971),
#  ("'O Guarani', um romance escrito por José de Alencar, é um marco do romantismo no Brasil e foi publicado em 1857.",
#   0.13183125469333265),
#  ("Clarice Lispector foi uma importante escritora brasileira, conhecida por obras como 'A Hora da Estrela' e 'Perto do Coração Selvagem'.",
#   0.05527551927638088)]
```
