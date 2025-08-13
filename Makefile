init:
	@poetry install
	@eval "$(poetry env activate)"

commit:
	@cz commit

pre-commit:
	@git add .
	@pre-commit run --all-files
