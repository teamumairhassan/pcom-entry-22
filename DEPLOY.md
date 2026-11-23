# Deploy on a VM and submit Homework 6

## A. Put the code in your GitHub repo and tag it
```bash
cd <your-repo>
# copy all files of this folder into your repo (keep the folder structure: data/, tests/)
git add .
git commit -m "Homework 6: prediction service (FastAPI, /health and /predict)"
git tag pcom-entry-2
git push origin HEAD --tags
```

## B. Create the VM (RRZE compute cloud, cc.rrze.de)
1. Create an Ubuntu VM (2 vCPU / 4 GB RAM is enough). Add your SSH key.
2. In the cloud security group, allow incoming **TCP port 8000** (and 22 for SSH).
3. Note the VM's public IP address.

## C. Install Docker and start the service (on the VM)
```bash
ssh ubuntu@<VM-IP>
sudo apt-get update && sudo apt-get install -y docker.io git
sudo systemctl enable --now docker
sudo ufw allow 8000/tcp        # only if ufw is active
git clone <your-repo-url> app && cd app
sudo docker build -t ml-prediction-api .        # trains the model inside the image (a few minutes)
sudo docker run -d --name ml-api --restart unless-stopped -p 8000:8000 ml-prediction-api
```
`--restart unless-stopped` also restarts it after a VM reboot.

## D. Check it
From the VM and from your own computer:
```bash
curl http://<VM-IP>:8000/health        # must answer {"health":"up"}
sudo docker logs -f ml-api             # shows each prediction request
```

## E. Register on Compy
1. Open https://compy.uni1.de, register, join **PCOM-01**, create your project.
2. Enter the VM address `http://<VM-IP>:8000` and your student id (pseudonym).
3. Rounds run every Monday at 4pm. The VM must be up at that time.

## Update later
```bash
cd app && git pull
sudo docker build -t ml-prediction-api . && sudo docker rm -f ml-api
sudo docker run -d --name ml-api --restart unless-stopped -p 8000:8000 ml-prediction-api
```
