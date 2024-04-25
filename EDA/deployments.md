# Deployment of Streamlit with Nginx

Deployment of Streamlit with Nginx involves several steps, including configuring Nginx as a reverse proxy for Streamlit, setting up a SystemD service for Streamlit, and managing Nginx configurations. Below is a step-by-step deployment guide:

## Step 1: Running Streamlit App

Run your Streamlit app using the following command in your terminal:

```sh
streamlit run app.py
```

# Step 2: Configuring Nginx

Create an Nginx configuration file for Streamlit by opening a new file with the following command:

```sh
nano /etc/nginx/conf.d/streamlit.conf
```


Add the following configuration to the streamlit.conf file:
```
upstream eda0 {
    server 127.0.0.1:8501; # This should be your Streamlit application running IP and port.
}

server {
    listen 80; # Port to listen on (remove "ssl" related lines if not using SSL)
    server_name da-tu.ca/eda; # Your domain name

    client_max_body_size 100M; # Maximum allowed file size

    location / {
        proxy_pass http://eda0;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $http_host;
        proxy_redirect off;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Save the file and exit the text editor.

# Step 3: Making Streamlit App a SystemD Service

Create a shell script for running your Streamlit app. Open a new file with the following command:

```sh
nano rundashboard.sh
```

Add the following lines to the rundashboard.sh script, replacing <prj folder> and <adr> with your project folder and address:

```sh
#!/bin/bash
cd /home/<prj folder>
source /<adr>/venv/bin/activate
nohup streamlit run dashboard.py
```

Move the shell script to the /usr/bin directory and make it executable:

```sh
sudo mv ./rundashboard.sh /usr/bin/rundashboard.sh
```


Create a SystemD unit file for your Streamlit service:

```sh
nano /etc/systemd/system/rundashboard.service
```

Add the following content to the rundashboard.service file:

```
[Unit]
Description=Run Streamlit Dashboard.

[Service]
Type=simple
ExecStart=/bin/bash /usr/bin/rundashboard.sh

[Install]
WantedBy=multi-user.target
```

Set the correct permissions for the unit file:

```sh
sudo chmod 644 /etc/systemd/system/rundashboard.service
```

Reload SystemD to recognize the new service and start it:

```sh
systemctl daemon-reload
systemctl start rundashboard
systemctl enable rundashboard
```

# Step 4: Starting Nginx and Managing Configurations

To start Nginx with the new Streamlit configuration, comment out the top server block in your Nginx configuration file (if necessary):

```sh
nano /etc/nginx/conf.d/flask.conf
```

Check the Nginx configuration for errors:

```sh
nginx -t
```

If there are no errors, restart Nginx to apply the changes:

```sh
systemctl restart nginx
```

# Step 5: Starting Streamlit Service

To start the Streamlit service, run the following commands:

```sh
systemctl start rundashboard
systemctl enable rundashboard
```

Check the status of the Streamlit service to ensure it's running:

```sh
systemctl status rundashboard
```

Your Streamlit app should now be deployed and accessible through Nginx using the specified domain and subdomain. 