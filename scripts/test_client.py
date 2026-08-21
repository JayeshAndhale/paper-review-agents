import arxiv
client = arxiv.Client()
result = next(client.results(arxiv.Search(id_list=["1706.03762"])))
print([a for a in dir(result) if not a.startswith('_')])