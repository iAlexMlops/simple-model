pipeline {
  agent {
    label 'jenkins/jenkins-jenkins-agent'
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