# HAAM_ros_pkg

ROS1 package for the Human Aware Andon Module

## Test

### Build

```bash
cd .docker
docker compose build
```

### Run

```
cd .docker
docker compose run --rm ros1
```

May need to make scripts executable:
```bash
chmod +x src/haam_ros_pkg/scripts/*
```


```bash
roslaunch haam_ros_pkg/launch/haam_nodes.launch
```


