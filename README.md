# Analisis sastra otomatis untuk bab buku (BookSum)

Kelompok 2A INT, mata kuliah Natural Language Processing 2024.

Dataset: [kmfoda/booksum](https://huggingface.co/datasets/kmfoda/booksum), berisi bab buku dan dua
teks rujukan per bab, yaitu `summary_text` (ringkasan alur) dan `summary_analysis` (analisis sastra).
Yang kami kerjakan adalah conditional text generation dengan arah `chapter` -> `summary_analysis`.
Kelompok 2B memakai dataset yang sama untuk summarization dengan target `summary_text`, jadi kedua
topik tidak tumpang tindih.

## Isi folder

```
booksum-colab.ipynb urutan pemanggilan skrip untuk dijalankan di Colab
src/config.py      semua parameter eksperimen dikumpulkan di sini
src/verify.py      cek kondisi dataset: nilai null, duplikasi bab, tumpang tindih antar split
src/data.py        memfilter baris dan menambah kolom chapter_selected
src/excerpt.py     seleksi kalimat berdasarkan TF-IDF dengan pembagian posisi bab
src/eda.py         mengukur panjang teks dalam token untuk tiap tokenizer
src/train.py       training tiga varian model
src/evaluate.py    ROUGE, BERTScore, dan perbandingan greedy dengan beam search
src/demo.py        antarmuka Gradio
outputs/           hasil antara dan hasil akhir, dibuat otomatis
```

## Menjalankan di Google Colab

Jalan pintasnya: buka `booksum-colab.ipynb` lewat
[Colab](https://colab.research.google.com/github/RayDitya12/booksum-literary-analysis/blob/main/booksum-colab.ipynb).
Notebook itu tidak berisi logika, hanya urutan pemanggilan skrip di `src/`, dan runtime GPU-nya sudah
dipasang lewat metadata notebook. Uraiannya di bawah, untuk kalau kamu lebih suka notebook sendiri.

Pilih Runtime -> Change runtime type -> T4 GPU sebelum menjalankan training.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/RayDitya12/booksum-literary-analysis.git
%cd booksum-literary-analysis
!pip install -q -r requirements.txt
```

Jalankan bertahap, satu sel untuk satu tahap:

```bash
!python src/verify.py
!python src/data.py
!python src/eda.py --limit 300
```

Uji dulu dengan sampel kecil sebelum training penuh, karena bagian ini yang paling sering
menimbulkan error pada versi library:

```bash
!python src/train.py --arm m1_bart_direct --limit 30 --epochs 0.05
!python src/evaluate.py --arm m1_bart_direct --limit 10
```

Training penuh, sebaiknya satu varian per sel:

```bash
!python src/train.py --arm m1_bart_direct
!python src/train.py --arm m2_bart_hybrid
!python src/train.py --arm m3_led_long
!python src/evaluate.py --arm m1_bart_direct
!python src/evaluate.py --arm m2_bart_hybrid
!python src/evaluate.py --arm m3_led_long
```

Sesi Colab gratis terputus setelah beberapa jam dan isi `/content` ikut hilang, jadi salin hasilnya
ke Drive setiap kali satu tahap selesai:

```bash
!cp -r outputs /content/drive/MyDrive/booksum-outputs
```

Model yang tersimpan di `outputs/<nama varian>` dipakai lagi oleh tahap evaluasi dan demo:

```bash
!python src/demo.py --arm m3_led_long
```

Sel demo akan terus berjalan dan menampilkan URL gradio.live di output sel tersebut.

## Menjalankan sebagian di laptop

`verify.py`, `data.py`, `excerpt.py`, dan `eda.py` hanya butuh CPU dan RAM biasa, jadi bisa dijalankan
lokal tanpa install torch:

```bash
python -m venv .venv
.venv/Scripts/pip install datasets transformers
.venv/Scripts/python src/data.py
.venv/Scripts/python src/eda.py --limit 300
```

Training dan evaluasi tidak realistis di laptop tanpa GPU, jadi bagian itu kami kerjakan di Colab.

## Hal yang perlu diperhatikan

Hasil `eda.py` (`outputs/token_stats.json`) menentukan nilai `max_src_tokens` di `config.py`, jadi
tahap itu dijalankan sebelum training, bukan sesudah.

Kolom `summary_text` tidak pernah dipakai sebagai input model. Kolom itu adalah target kelompok 2B
dan isinya merangkum alur bab yang sama, sehingga memasukkannya ke input membuat model melihat
informasi yang seharusnya ia hasilkan sendiri.

Angka pembanding antar varian diambil dari `outputs/leaderboard.jsonl`, yang ditulis oleh
`evaluate.py` setiap kali satu varian dievaluasi.
