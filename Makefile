init:
	@poetry install
	@eval "$(poetry env activate)"

commit:
	@cz commit

pre-commit:
	@pre-commit run
