# Setup on Windows 11 (WSL2) - Ryzen 5 5500U, 16 GB RAM, no NVIDIA GPU

Everything runs on CPU inside WSL2 Ubuntu 22.04.

1. PowerShell (admin): `wsl --install -d Ubuntu-22.04`, reboot.
2. `C:\Users\<you>\.wslconfig`:
   ```
   [wsl2]
   memory=10GB
   processors=8
   swap=8GB
   ```
   then `wsl --shutdown`.
3. Install **Docker Desktop** (WSL2 backend, enable Ubuntu integration).
4. Install **Cursor**, add the WSL extension, "Connect to WSL", open the repo folder inside `~/`.
5. In Ubuntu: `sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip openjdk-17-jdk git make`
   (use the deadsnakes PPA if python3.10 is missing).
6. `git clone <your repo> && cd recsys-two-stage && make setup && source .venv/bin/activate`

Keep the repo inside the WSL filesystem (`~/`), not `/mnt/c`, for 5-10× faster I/O.

## Memory tips
- Close Chrome while training; use `MODE=sample` (20% of users) for iteration.
- Spark driver is capped at 6 GB (`spark.driver_memory`). Lower `ranking.max_train_users` if the ranker set OOMs.
- Start Docker services only **after** training (`make train export`), they need ~9 GB.
