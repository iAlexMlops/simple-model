pipeline {
  agent {
    kubernetes {
      inheritFrom 'jenkins-agent-pod'
      yaml '''
      spec:
        containers:
        - name: docker
          image: docker:dind
'''
    }
  }
  stages {
    stage('Build & Push Image') {
      steps {
        container('docker') {
          script {
            sh "docker buildx build -t ialexmlops/alex-test-docker:1.0"
          }
        }
      }
    }
  }
}