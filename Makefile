#!make
include .env
export

build-image-local:
	docker build -t felixpeters/ml-env:latest .

update-image-local:build-image-local
	docker push felixpeters/ml-env:latest

update-image-prod:
	gradient jobs create --name docker-ml-env --apiKey $(GRADIENT_API_KEY) --workspace https://github.com/felixpeters/ml-experiments.git --useDockerfile true --command "echo hello" --registryTarget felixpeters/ml-env:latest --registryTargetUsername $(DOCKER_REGISTRY_USER) --registryTargetPassword $(DOCKER_REGISTRY_PASSWORD) --machineType GPU+ --projectId $(GRADIENT_PROJECT_ID)
