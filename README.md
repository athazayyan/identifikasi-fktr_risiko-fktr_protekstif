# 🧠 Pipeline Penelitian PKM-RSH 2026

**Identifikasi Faktor Risiko & Protektif Kesehatan Mental Perempuan Penyintas Banjir Aceh melalui Indikator Linguistik Berbasis Machine Learning**

Universitas Syiah Kuala | Tim PKM-RSH 2026

---

## 📋 Tentang Aplikasi Ini

Aplikasi Streamlit interaktif ini memvisualisasikan **seluruh pipeline penelitian** dari proposal PKM-RSH, mencakup:

- **Fase 1** — Adaptasi instrumen DASS-21 & MSPSS ke wawancara naratif
- **Fase 2** — Data dummy wawancara 5 responden lengkap dengan transkrip
- **Fase 3** — Pipeline pra-pemrosesan NLP (transkripsi → POS tagging → stemming)
- **Fase 4** — Mekanisme pelabelan ground truth oleh tim peneliti
- **Fase 5A** — Eksplorasi kebahasaan: LIWC + BERTopic + Korelasi Spearman
- **Fase 5B** — Pemodelan IndoBERT + strategi dataset kecil N=20
- **Fase 6** — Feature fusion & peta risiko–protektif 2D
- **Fase 7** — Contoh hasil klasifikasi end-to-end

## 🚀 Cara Deploy ke Streamlit Cloud

1. Fork/clone repository ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect dengan akun GitHub
4. Pilih repository ini, file `app.py`
5. Deploy!

## 💻 Cara Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👥 Tim Peneliti

| Nama | Program Studi | Peran |
|------|--------------|-------|
| Cut Alvi Khairanda | Psikologi | Ketua Tim |
| Achmad Atha Zayyan | Informatika | Anggota (Teknis ML) |
| Durratul Irfana | Psikologi | Anggota (Data & Media) |

**Dosen Pendamping:** Dr. Haiyun Nisa, S.Psi., M.Psi., Psikolog

## ⚠️ Catatan

Data yang ditampilkan adalah **data dummy** untuk keperluan ilustrasi pipeline penelitian. Data responden aktual bersifat rahasia dan telah dianonimisasi.
