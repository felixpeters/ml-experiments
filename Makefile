build-image:
	docker build -t felixpeters/ml-env:latest .

update-image:build-image
	docker push felixpeters/ml-env:latest
