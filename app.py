import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Penelitian PKM-RSH | Mental Health NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2rem; font-weight:700; color:#1e3a5f; margin-bottom:0.2rem}
    .sub-title  {font-size:1rem; color:#555; margin-bottom:1.5rem}
    .phase-header {background:linear-gradient(90deg,#1e3a5f,#2d6fa4);
                   color:white; padding:0.7rem 1.2rem; border-radius:10px;
                   font-size:1.1rem; font-weight:600; margin-bottom:1rem}
    .info-box   {background:#eef4fb; border-left:4px solid #2d6fa4;
                 padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0}
    .warn-box   {background:#fff8e6; border-left:4px solid #f0a500;
                 padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0}
    .success-box{background:#edfaf1; border-left:4px solid #27ae60;
                 padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0}
    .danger-box {background:#fdf0f0; border-left:4px solid #c0392b;
                 padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin:0.5rem 0}
    .step-badge {background:#2d6fa4; color:white; padding:2px 10px;
                 border-radius:12px; font-size:0.78rem; font-weight:600}
    .card       {background:white; border:1px solid #e0e6ef; border-radius:10px;
                 padding:1rem 1.2rem; margin:0.5rem 0;
                 box-shadow:0 1px 4px rgba(0,0,0,0.06)}
    .narasi-block {background:#f8f9fb; border:1px solid #d0dce8; border-radius:8px;
                   padding:1rem; font-style:italic; color:#333; margin:0.4rem 0;
                   font-size:0.9rem; line-height:1.7}
    .label-risiko-tinggi  {background:#fdecea;color:#c0392b;padding:3px 10px;
                           border-radius:12px;font-size:0.78rem;font-weight:700}
    .label-risiko-sedang  {background:#fff3cd;color:#856404;padding:3px 10px;
                           border-radius:12px;font-size:0.78rem;font-weight:700}
    .label-risiko-rendah  {background:#d4edda;color:#155724;padding:3px 10px;
                           border-radius:12px;font-size:0.78rem;font-weight:700}
    .token-pron   {background:#fdecea;color:#c0392b;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .token-afekN  {background:#fdecea;color:#922b21;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .token-afekP  {background:#d4edda;color:#186a3b;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .token-sosial {background:#d6eaf8;color:#1a5276;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .token-rel    {background:#e8daef;color:#6c3483;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .token-neg    {background:#fdf2e9;color:#784212;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .token-normal {background:#f0f0f0;color:#555;padding:1px 6px;
                   border-radius:4px;font-size:0.82rem;margin:1px;display:inline-block}
    .wawancara-q  {color:#1e3a5f;font-weight:600;font-size:0.88rem;margin-top:0.8rem}
    .wawancara-a  {color:#333;font-size:0.88rem;line-height:1.7;margin-left:1rem;
                   border-left:3px solid #2d6fa4;padding-left:0.7rem;margin-top:0.2rem}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA DUMMY — 5 RESPONDEN LENGKAP
# ─────────────────────────────────────────────

WAWANCARA_DUMMY = {
    "R-001": {
        "nama": "Ibu Nurhayati (R-001)",
        "usia": 38,
        "desa": "Desa Meureudu, Pidie Jaya",
        "durasi": "67 menit",
        "bahasa": "Indonesia campur sedikit Aceh",
        "wawancara": [
            {
                "dimensi": "Rapport",
                "pertanyaan": "Bisa diceritakan sedikit tentang diri Ibu dan keseharian sebelum banjir?",
                "jawaban": "Saya Nurhayati, tinggal di sini sudah dua puluh tahun lebih. Suami kerja di Banda Aceh, pulang seminggu sekali. Saya ibu rumah tangga, ngurus tiga anak, yang paling kecil masih enam tahun."
            },
            {
                "dimensi": "Depresi",
                "pertanyaan": "Setelah banjir terjadi, bagaimana perubahan perasaan yang Ibu rasakan? Adakah aktivitas yang dulu mudah, sekarang terasa sulit?",
                "jawaban": "Ee... susah jelasinnya. Saya nggak bisa tidur, setiap malam kepikiran terus. Yang dulu saya suka masak buat anak-anak, sekarang... nggak ada semangat. Rasanya hampa gitu, Kak. Kayak nggak ada gunanya saya bangun pagi. Sering nangis sendiri di kamar, tapi nggak mau kelihatan sama anak-anak."
            },
            {
                "dimensi": "Kecemasan",
                "pertanyaan": "Apakah Ibu pernah merasa khawatir banjir akan datang lagi? Situasi apa yang biasanya membuat cemas?",
                "jawaban": "Takut banget, Kak. Setiap hujan lebat, saya langsung was-was, jantung berdegup kencang. Anak-anak tidur saya pikir terus, gimana kalau air naik lagi tengah malam. Saya sampai naruh tas darurat di dekat pintu, tiap malam periksa. Tidur pun nggak tenang, mimpi banjir terus."
            },
            {
                "dimensi": "Stres",
                "pertanyaan": "Hal apa yang paling membuat Ibu merasa tertekan atau terbebani saat ini?",
                "jawaban": "Capek banget, Kak. Suami jauh, jadi saya yang ngurusin semuanya — benerin rumah yang rusak, ngurus anak, cari nafkah seadanya. Saya ngerasa sendirian. Marah-marah sama anak kadang, terus nyesel sendiri. Semuanya saya yang tanggung."
            },
            {
                "dimensi": "Protektif — Keluarga",
                "pertanyaan": "Bagaimana peran keluarga dalam membantu Ibu menjalani hari-hari setelah banjir?",
                "jawaban": "Suami jauh, jadi ya sedikit. Dia transfer uang tapi nggak bisa ada di sini. Mertua sudah tua, nggak bisa banyak bantu. Kakak saya di Banda Aceh sempat datang tiga hari, itu lumayan."
            },
            {
                "dimensi": "Protektif — Teman/Komunitas",
                "pertanyaan": "Apakah ada teman atau tetangga yang membantu setelah banjir? Bagaimana hubungan Ibu dengan komunitas sekitar?",
                "jawaban": "Tetangga ada yang bantu beresin lumpur, tapi ya seadanya karena mereka juga kena banjir. Kelompok pengajian kami sempat kumpul, itu lumayan bikin lega bisa cerita sama sesama."
            },
            {
                "dimensi": "Protektif — Significant Others",
                "pertanyaan": "Adakah pihak yang sangat berarti bagi Ibu selama masa ini — bisa siapa saja?",
                "jawaban": "Ustazah di pengajian kami, beliau sering telepon nanya kabar. Kata beliau, ini ujian dari Allah dan kita harus sabar. Itu agak bikin tenang sedikit."
            }
        ],
        "label": {
            "depresi": "Moderate",       # D2
            "kecemasan": "Moderate",     # A2
            "stres": "Severe",           # S3
            "risiko_overall": "TINGGI",
            "protektif_keluarga": "Low",
            "protektif_teman": "Moderate",
            "protektif_so": "Moderate",
            "protektif_overall": "SEDANG"
        },
        "teks_bersih": "tidak bisa tidur malam kepikiran terus nggak ada semangat hampa nggak ada gunanya nangis sendiri takut banjir lagi jantung berdegup kencang was-was mimpi banjir terus capek sendirian ngurusin semua marah nyesel suami jauh saya saya saya",
        "skor_risiko_num": 3,
        "skor_protektif_num": 2,
    },

    "R-002": {
        "nama": "Ibu Marlina (R-002)",
        "usia": 44,
        "desa": "Desa Trienggadeng, Pidie Jaya",
        "durasi": "52 menit",
        "bahasa": "Indonesia",
        "wawancara": [
            {
                "dimensi": "Rapport",
                "pertanyaan": "Bisa diceritakan tentang diri Ibu dan kegiatan sehari-hari?",
                "jawaban": "Saya Marlina, pedagang di pasar. Suami petani, kami punya empat anak. Sudah hampir tiga puluh tahun di sini."
            },
            {
                "dimensi": "Depresi",
                "pertanyaan": "Bagaimana perasaan Ibu setelah banjir? Ada yang berubah dari keseharian Ibu?",
                "jawaban": "Alhamdulillah, kami bersyukur masih selamat semua. Memang berat di awal, tapi kami coba bangkit bersama. Saya masih berjualan meski barang banyak yang rusak. Anak-anak bantu beresin rumah, kami kerjakan bersama-sama."
            },
            {
                "dimensi": "Kecemasan",
                "pertanyaan": "Apakah Ibu masih sering merasa khawatir akan banjir lagi?",
                "jawaban": "Ada rasa khawatir, itu manusiawi. Tapi kami sudah pasrah sama Allah. Kalau hujan lebat ya kami waspada, tapi tidak sampai panik. Kami sudah buat kesepakatan sama tetangga, kalau ada tanda bahaya langsung saling info."
            },
            {
                "dimensi": "Stres",
                "pertanyaan": "Hal apa yang paling berat dirasakan sekarang?",
                "jawaban": "Modal usaha habis, itu yang berat. Tapi suami dan anak-anak kompak bantu. Kami coba ajukan ke bantuan pemerintah. Berat memang, tapi kami hadapi bersama, tidak sendirian."
            },
            {
                "dimensi": "Protektif — Keluarga",
                "pertanyaan": "Bagaimana peran keluarga dalam masa pemulihan ini?",
                "jawaban": "Suami sangat mendukung, anak-anak juga. Kami selalu musyawarah mau ngapain. Keluarga besar dari kampung juga datang bantu-bantu. Kami tidak sendirian, itu yang bikin kuat."
            },
            {
                "dimensi": "Protektif — Teman/Komunitas",
                "pertanyaan": "Bagaimana dukungan dari teman dan komunitas?",
                "jawaban": "Komunitas kami solid. Sesama pedagang pasar saling bantu. Ada yang pinjamkan modal, ada yang bagi bahan makanan. Gotong royong itu masih kuat di sini. Kami merasa satu senasib sepenanggungan."
            },
            {
                "dimensi": "Protektif — Significant Others",
                "pertanyaan": "Adakah pihak yang paling berarti bagi Ibu selama masa ini?",
                "jawaban": "Pak Keuchik kami sangat aktif koordinir bantuan. Dan pengajian rutin tetap jalan meski kondisi susah. Itu penting banget buat kami — bisa ketemu, bisa saling menguatkan, berdoa bersama."
            }
        ],
        "label": {
            "depresi": "Normal",
            "kecemasan": "Mild",
            "stres": "Mild",
            "risiko_overall": "RENDAH",
            "protektif_keluarga": "High",
            "protektif_teman": "High",
            "protektif_so": "High",
            "protektif_overall": "TINGGI"
        },
        "teks_bersih": "bersyukur masih selamat bangkit bersama kerjakan bersama-sama pasrah Allah waspada saling info kompak bersama gotong royong solid saling bantu satu senasib menguatkan berdoa bersama kami kami kami kita",
        "skor_risiko_num": 1,
        "skor_protektif_num": 3,
    },

    "R-003": {
        "nama": "Ibu Rahmawati (R-003)",
        "usia": 29,
        "desa": "Desa Ulim, Pidie Jaya",
        "durasi": "81 menit",
        "bahasa": "Aceh dominan, diterjemahkan",
        "wawancara": [
            {
                "dimensi": "Rapport",
                "pertanyaan": "Bisa Ibu ceritakan sedikit tentang diri dan keseharian?",
                "jawaban": "[Terjemahan dari Bahasa Aceh] Saya masih muda, baru menikah dua tahun. Suami kerja serabutan. Kami belum punya anak. Tinggal di rumah kontrakan yang sekarang rusak kena banjir."
            },
            {
                "dimensi": "Depresi",
                "pertanyaan": "Bagaimana perubahan perasaan setelah banjir? Adakah hal yang dulu bisa dilakukan tapi sekarang sulit?",
                "jawaban": "[Terjemahan] Saya tidak tahu harus bagaimana. Rasanya gelap sekali. Tidak ada yang bisa saya harapkan. Saya tidak mau ngapa-ngapain, males keluar rumah, males ngomong sama suami pun. Setiap hari nangis, tapi nggak tahu kenapa juga. Badan pun rasanya berat, males gerak."
            },
            {
                "dimensi": "Kecemasan",
                "pertanyaan": "Apakah ada kekhawatiran yang terus menghantui sejak banjir?",
                "jawaban": "[Terjemahan] Sangat takut. Setiap ada suara hujan, saya langsung gemetar, nafas sesak. Malam-malam sering terbangun karena mimpi buruk, teriak dalam tidur kata suami saya. Saya takut sendirian di rumah kalau suami kerja."
            },
            {
                "dimensi": "Stres",
                "pertanyaan": "Apa yang paling berat ditanggung sekarang?",
                "jawaban": "[Terjemahan] Rumah kontrakan rusak, kami nggak punya uang buat pindah atau benerin. Suami nggak bisa kerja karena juga kena banjir di tempat kerjanya. Kami makan seadanya dari bantuan. Saya ngerasa tidak berguna, tidak bisa bantu suami."
            },
            {
                "dimensi": "Protektif — Keluarga",
                "pertanyaan": "Bagaimana dukungan dari keluarga?",
                "jawaban": "[Terjemahan] Keluarga saya di Bireuen, jauh. Orang tua sudah tahu tapi tidak bisa ke sini. Mertua juga kena banjir. Suami ada tapi dia juga stress, jadi kami malah sering diam-diaman."
            },
            {
                "dimensi": "Protektif — Teman/Komunitas",
                "pertanyaan": "Apakah ada teman atau tetangga yang membantu?",
                "jawaban": "[Terjemahan] Di sini kami pendatang, belum banyak kenal. Tetangga ada kasih makanan sekali dua kali, tapi tidak dekat. Saya tidak tahu harus minta tolong ke siapa."
            },
            {
                "dimensi": "Protektif — Significant Others",
                "pertanyaan": "Adakah siapa pun yang terasa sangat berarti dan menguatkan Ibu saat ini?",
                "jawaban": "[Terjemahan] Hm... ada relawan perempuan yang datang dari LSM. Dia sempat ngobrol lama sama saya. Itu satu-satunya yang bikin saya ngerasa didengar. Tapi dia sudah pergi setelah seminggu."
            }
        ],
        "label": {
            "depresi": "Severe",
            "kecemasan": "Severe",
            "stres": "Severe",
            "risiko_overall": "TINGGI",
            "protektif_keluarga": "Low",
            "protektif_teman": "Low",
            "protektif_so": "Low",
            "protektif_overall": "RENDAH"
        },
        "teks_bersih": "tidak tahu bagaimana gelap tidak ada harapan tidak mau apa-apa nangis tidak tahu kenapa badan berat gemetar nafas sesak mimpi buruk teriak takut sendirian tidak berguna tidak bisa bantu diam-diaman tidak kenal tidak tahu minta tolong saya saya saya aku",
        "skor_risiko_num": 3,
        "skor_protektif_num": 1,
    },

    "R-004": {
        "nama": "Ibu Suryani (R-004)",
        "usia": 51,
        "desa": "Desa Jangka Buya, Pidie Jaya",
        "durasi": "58 menit",
        "bahasa": "Indonesia",
        "wawancara": [
            {
                "dimensi": "Rapport",
                "pertanyaan": "Bisa diceritakan tentang diri Ibu?",
                "jawaban": "Saya Suryani, guru SD sudah dua puluh tahun. Suami sudah meninggal lima tahun lalu. Saya tinggal sama anak sulung dan cucunya."
            },
            {
                "dimensi": "Depresi",
                "pertanyaan": "Bagaimana perasaan Ibu pasca-banjir? Ada aktivitas yang terasa berbeda?",
                "jawaban": "Sedih iya, banyak kenangan yang hilang. Tapi saya coba tetap mengajar meski susah. Anak-anak butuh guru mereka. Itu yang bikin saya bangkit — kalau saya jatuh, siapa yang urus mereka? Saya nangis sekali tapi ya sudah, hidup terus jalan."
            },
            {
                "dimensi": "Kecemasan",
                "pertanyaan": "Apakah Ibu masih sering merasa cemas?",
                "jawaban": "Ada was-was kalau hujan deras. Tapi saya coba kendalikan. Saya bilang ke diri sendiri, kita sudah lewati ini dan kita bisa lewati lagi. Doa juga yang menguatkan saya."
            },
            {
                "dimensi": "Stres",
                "pertanyaan": "Apa yang paling berat saat ini?",
                "jawaban": "Kasihan murid-murid yang sekolahnya rusak. Saya ikut bantu koordinir kelas darurat. Capek iya, tapi ada kepuasan. Saya tidak bisa diam saja kalau ada yang butuh bantuan."
            },
            {
                "dimensi": "Protektif — Keluarga",
                "pertanyaan": "Bagaimana peran keluarga dalam pemulihan?",
                "jawaban": "Anak saya sangat peduli, dia yang urus semua keperluan rumah. Saudara-saudara juga datang bantu. Kami keluarga besar, selalu kompak. Alhamdulillah tidak pernah merasa sendirian."
            },
            {
                "dimensi": "Protektif — Teman/Komunitas",
                "pertanyaan": "Bagaimana dukungan dari rekan dan komunitas?",
                "jawaban": "Sesama guru sangat solid, kami bergantian mengajar di kelas darurat. Wali murid juga bantu logistik. Komunitas ini yang bikin saya kuat, kami saling bergantung satu sama lain."
            },
            {
                "dimensi": "Protektif — Significant Others",
                "pertanyaan": "Siapa yang paling menguatkan Ibu dalam masa ini?",
                "jawaban": "Kepala sekolah saya sangat suportif. Dan tentu saja doa — saya rutin ke majelis taklim, di sana kami saling mendoakan, itu energi yang luar biasa. Allah tidak akan membebani melebihi kemampuan kita."
            }
        ],
        "label": {
            "depresi": "Mild",
            "kecemasan": "Mild",
            "stres": "Moderate",
            "risiko_overall": "SEDANG",
            "protektif_keluarga": "High",
            "protektif_teman": "High",
            "protektif_so": "High",
            "protektif_overall": "TINGGI"
        },
        "teks_bersih": "sedih tapi coba bangkit mengajar hidup terus jalan was-was tapi kendalikan doa menguatkan capek tapi kepuasan kompak tidak sendirian solid saling bergantung saling mendoakan energi Allah tidak membebani kami kami kita",
        "skor_risiko_num": 2,
        "skor_protektif_num": 3,
    },

    "R-005": {
        "nama": "Ibu Fatimah (R-005)",
        "usia": 35,
        "desa": "Desa Panteraja, Pidie Jaya",
        "durasi": "74 menit",
        "bahasa": "Indonesia campur Aceh",
        "wawancara": [
            {
                "dimensi": "Rapport",
                "pertanyaan": "Bisa diceritakan tentang diri Ibu dan keseharian?",
                "jawaban": "Saya Fatimah, petani kecil-kecilan. Suami juga petani. Kami punya dua anak yang masih SD. Tinggal di sini dari lahir."
            },
            {
                "dimensi": "Depresi",
                "pertanyaan": "Bagaimana perubahan perasaan Ibu setelah banjir?",
                "jawaban": "Susah... ee, ya memang susah. Sawah kami habis, modal tanam habis. Saya sedih banget, kerja keras berbulan-bulan hilang sekejap. Kadang ngerasa nggak semangat, tapi lihat anak-anak ya harus bangkit. Mereka butuh saya."
            },
            {
                "dimensi": "Kecemasan",
                "pertanyaan": "Apakah ada kekhawatiran yang sering muncul?",
                "jawaban": "Khawatir soal sekolah anak-anak, uang sekolah gimana. Kalau hujan deras saya was-was. Tapi saya coba tanamkan ke diri sendiri, bersyukur masih hidup dan sehat."
            },
            {
                "dimensi": "Stres",
                "pertanyaan": "Apa yang paling berat?",
                "jawaban": "Keuangan itu paling berat. Kami dari bantuan seadanya. Suami coba cari kerja lain sementara, saya juga bantu jualan kecil-kecilan. Berat tapi ya kami jalani bersama."
            },
            {
                "dimensi": "Protektif — Keluarga",
                "pertanyaan": "Bagaimana peran keluarga?",
                "jawaban": "Suami sangat suportif, kami saling kuat-kuatin. Anak-anak walau kecil sudah ngerti situasi, mereka tidak banyak minta. Itu yang bikin kami makin kompak."
            },
            {
                "dimensi": "Protektif — Teman/Komunitas",
                "pertanyaan": "Bagaimana dukungan komunitas?",
                "jawaban": "Kelompok tani kami masih aktif. Ada yang pinjamkan benih buat tanam lagi. Tetangga juga saling bantu, kami gotong royong beresin sawah bareng. Senang ada orang-orang seperti itu."
            },
            {
                "dimensi": "Protektif — Significant Others",
                "pertanyaan": "Siapa yang paling berarti?",
                "jawaban": "Suami nomor satu. Terus ada penyuluh pertanian yang datang bantu kami rencanakan tanam ulang. Itu sangat membantu secara praktis. Dan doa, selalu doa — itu yang bikin tenang."
            }
        ],
        "label": {
            "depresi": "Mild",
            "kecemasan": "Mild",
            "stres": "Moderate",
            "risiko_overall": "SEDANG",
            "protektif_keluarga": "High",
            "protektif_teman": "Moderate",
            "protektif_so": "Moderate",
            "protektif_overall": "SEDANG"
        },
        "teks_bersih": "susah sedih habis nggak semangat tapi bangkit bersyukur masih hidup was-was tapi bersyukur bersama jalani bersama saling kuat-kuatin kompak gotong royong saling bantu doa tenang suami kami kami",
        "skor_risiko_num": 2,
        "skor_protektif_num": 2,
    },
}

# ─────────────────────────────────────────────
# KAMUS LINGUISTIK
# ─────────────────────────────────────────────

KAMUS = {
    "afek_negatif": ["takut", "was-was", "cemas", "sedih", "hampa", "putus asa", "gelap",
                     "nangis", "menyesal", "nyesel", "gemetar", "sesak", "panik",
                     "khawatir", "susah", "berat", "lelah", "capek", "stress", "marah",
                     "tertekan", "tidak berguna", "tidak berharga", "tidak ada gunanya",
                     "mimpi buruk", "tidak tenang", "kewalahan"],
    "afek_positif": ["bersyukur", "senang", "lega", "tenang", "bangkit", "kuat", "bahagia",
                     "semangat", "positif", "harapan", "alhamdulillah", "sabar",
                     "ikhlas", "pasrah", "kepuasan", "baik", "bahagia"],
    "pron_tunggal": ["saya", "aku", "ku", "ku-", "-ku", "diri saya", "diri sendiri"],
    "pron_jamak":   ["kami", "kita", "kita semua", "bersama", "bersama-sama"],
    "sosial":       ["keluarga", "suami", "anak", "tetangga", "komunitas", "teman",
                     "saudara", "mertua", "orang tua", "relawan", "gotong royong",
                     "bersama", "solid", "bantu", "saling", "musyawarah"],
    "religius":     ["allah", "doa", "berdoa", "shalat", "ibadah", "bersyukur", "ikhlas",
                     "pasrah", "ujian", "sabar", "rezeki", "tawakal", "alhamdulillah",
                     "insyaallah", "majelis", "pengajian", "ustaz", "ustazah"],
    "negasi":       ["tidak", "nggak", "tak", "bukan", "belum", "jangan", "tanpa",
                     "tiada", "nihil", "nggak ada"],
    "isolasi":      ["sendirian", "sendiri", "terisolir", "tidak kenal", "tidak ada yang",
                     "diam", "diam-diaman", "jauh", "terpisah"],
    "pasif_impoten":["tidak bisa", "tidak mampu", "tidak berdaya", "mau gimana lagi",
                     "tidak tahu harus bagaimana", "tidak berguna", "tidak berharga"],
}


def hitung_liwc(teks: str) -> dict:
    """Hitung frekuensi fitur linguistik dari teks bersih."""
    teks_lower = teks.lower()
    kata_list = re.findall(r'\b\w+\b', teks_lower)
    total = len(kata_list) if kata_list else 1
    hasil = {}
    for kategori, kata_kunci in KAMUS.items():
        count = sum(teks_lower.count(k) for k in kata_kunci)
        hasil[kategori] = round((count / total) * 100, 2)
    return hasil


def warnai_token(teks: str) -> str:
    """Buat HTML teks berewarnai sesuai kategori linguistik."""
    kata_list = teks.split()
    html_parts = []
    for kata in kata_list:
        kata_lower = kata.lower().strip(".,!?;:")
        cls = "token-normal"
        if any(kata_lower == k or k in kata_lower for k in KAMUS["pron_tunggal"]):
            cls = "token-pron"
        elif any(kata_lower == k or k in kata_lower for k in KAMUS["afek_negatif"]):
            cls = "token-afekN"
        elif any(kata_lower == k or k in kata_lower for k in KAMUS["afek_positif"]):
            cls = "token-afekP"
        elif any(kata_lower == k or k in kata_lower for k in KAMUS["sosial"]):
            cls = "token-sosial"
        elif any(kata_lower == k or k in kata_lower for k in KAMUS["religius"]):
            cls = "token-rel"
        elif any(kata_lower == k or k in kata_lower for k in KAMUS["negasi"]):
            cls = "token-neg"
        html_parts.append(f'<span class="{cls}">{kata}</span>')
    return " ".join(html_parts)


def pos_tag_dummy(teks: str) -> list:
    """Simulasi POS tagging untuk teks bersih."""
    hasil = []
    for kata in teks.split():
        k = kata.lower().strip(".,!?")
        if k in KAMUS["pron_tunggal"]:
            pos, keterangan = "PRON-SG", "Kata ganti orang pertama tunggal"
        elif k in KAMUS["pron_jamak"]:
            pos, keterangan = "PRON-PL", "Kata ganti orang pertama jamak"
        elif k in KAMUS["afek_negatif"]:
            pos, keterangan = "ADJ/V-NEG", "Kata afek negatif"
        elif k in KAMUS["afek_positif"]:
            pos, keterangan = "ADJ/V-POS", "Kata afek positif"
        elif k in KAMUS["negasi"]:
            pos, keterangan = "NEG", "Kata negasi"
        elif k in KAMUS["sosial"]:
            pos, keterangan = "NOUN-SOC", "Kata sosial/relasi"
        elif k in KAMUS["religius"]:
            pos, keterangan = "NOUN-REL", "Kata religius"
        elif k in KAMUS["isolasi"]:
            pos, keterangan = "ADJ-ISO", "Kata isolasi/kesendirian"
        elif k in ["banjir", "air", "hujan", "lumpur", "rumah", "sawah", "kontrakan"]:
            pos, keterangan = "NOUN", "Kata benda konteks bencana"
        else:
            pos, keterangan = "MISC", "Lainnya"
        hasil.append({"Token": kata, "POS": pos, "Keterangan": keterangan})
    return hasil


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Pipeline Penelitian PKM-RSH")
    st.markdown("**Identifikasi Faktor Risiko & Protektif Kesehatan Mental Perempuan Penyintas Banjir di Aceh melalui Indikator Linguistik**")
    st.divider()

    fase = st.radio("📂 Pilih Fase:", [
        "🏠 Gambaran Umum",
        "📋 Fase 1 — Persiapan & Instrumen",
        "🎤 Fase 2 — Pengumpulan Data",
        "🔧 Fase 3 — Pra-pemrosesan",
        "🏷️ Fase 4 — Pelabelan Ground Truth",
        "🔍 Fase 5A — Eksplorasi Kebahasaan",
        "🤖 Fase 5B — Pemodelan IndoBERT",
        "🗺️ Fase 6 — Sintesis & Pemetaan",
        "📄 Fase 7 — Contoh Hasil Klasifikasi",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Legenda Warna Token:**")
    st.markdown("""
    <span class="token-pron">PRON-I tunggal</span> → Risiko depresi<br>
    <span class="token-afekN">Afek negatif</span> → Risiko<br>
    <span class="token-afekP">Afek positif</span> → Protektif<br>
    <span class="token-sosial">Kata sosial</span> → Protektif<br>
    <span class="token-rel">Kata religius</span> → Protektif<br>
    <span class="token-neg">Negasi</span> → Perlu perhatian
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("Universitas Syiah Kuala · PKM-RSH 2026")


# ─────────────────────────────────────────────
# FASE 0 — GAMBARAN UMUM
# ─────────────────────────────────────────────
if fase == "🏠 Gambaran Umum":
    st.markdown('<div class="main-title">🧠 Pipeline Penelitian PKM-RSH</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Identifikasi Faktor Risiko & Protektif Kesehatan Mental Perempuan Penyintas Banjir Aceh melalui Indikator Linguistik Berbasis Machine Learning</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👩 Responden", "20 perempuan")
    col2.metric("📍 Lokasi", "Pidie Jaya, Aceh")
    col3.metric("🤖 Model", "IndoBERT")
    col4.metric("📊 Evaluasi", "Macro F1-Score")

    st.divider()
    st.markdown("### 🔄 Alur Penelitian Lengkap")

    tahapan = [
        ("1", "Persiapan", "Studi literatur, adaptasi DASS-21 & MSPSS ke wawancara, uji etik, pilot study 5 responden", "#6c757d"),
        ("2", "Pengumpulan Data", "Wawancara naratif mendalam 20 perempuan penyintas banjir di Pidie Jaya (~60-90 menit/orang)", "#2d6fa4"),
        ("3", "Pra-pemrosesan", "Transkripsi verbatim → normalisasi → tokenisasi → POS tagging → stopword removal → stemming", "#17a589"),
        ("4", "Pelabelan", "Tim peneliti menilai transkrip berdasarkan rubrik DASS-21 & MSPSS → ground truth label risiko & protektif", "#8e44ad"),
        ("5A", "Eksplorasi Kebahasaan", "LIWC (frekuensi kategori kata) + BERTopic (tema laten) + Spearman correlation", "#d4a017"),
        ("5B", "Pemodelan IndoBERT", "Fine-tuning IndoBERT → klasifikasi risiko & protektif → LOO-CV evaluasi", "#c0392b"),
        ("6", "Sintesis", "Feature fusion LIWC + IndoBERT → peta risiko–protektif 2D per responden", "#27ae60"),
        ("7", "Luaran", "Dataset anotasi + Model HKI + Artikel SINTA + Akun medsos edukasi", "#34495e"),
    ]

    for t in tahapan:
        with st.container():
            c1, c2 = st.columns([1, 11])
            with c1:
                st.markdown(f"""<div style="background:{t[3]};color:white;width:44px;height:44px;
                border-radius:50%;display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:1rem;margin-top:4px">{t[0]}</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{t[1]}**")
                st.caption(t[2])

    st.divider()
    st.markdown("### 🔑 Inovasi Utama Penelitian Ini")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-box"><strong>📝 Adaptasi Instrumen</strong><br>DASS-21 & MSPSS ditranslasi menjadi pertanyaan wawancara naratif yang lebih manusiawi untuk penyintas bencana. Tim peneliti yang melabeli — bukan responden yang mengisi kuesioner.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="success-box"><strong>🌏 Konteks Lokal Aceh</strong><br>Kamus linguistik diadaptasi untuk Bahasa Indonesia dialek Aceh. Menangkap ekspresi lokal seperti "lon" (saya), "meunan" (seperti itu), dan coping religius khas budaya Aceh.</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="warn-box"><strong>🔗 Interdisipliner</strong><br>Psikologi Klinis (DASS-21/MSPSS) + Linguistik Komputasional (LIWC/BERTopic) + Deep Learning (IndoBERT) — ketiganya terhubung organik, bukan sekadar ditempel.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FASE 1 — PERSIAPAN
# ─────────────────────────────────────────────
elif fase == "📋 Fase 1 — Persiapan & Instrumen":
    st.markdown('<div class="phase-header">📋 Fase 1 — Persiapan & Instrumen</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["1.1 Instrumen Asal", "1.2 Adaptasi ke Wawancara", "1.3 Uji Etik & Pilot"])

    with tab1:
        st.markdown("#### Instrumen Asal: DASS-21 & MSPSS")
        st.markdown('<div class="info-box">DASS-21 dan MSPSS adalah instrumen psikometri tervalidasi secara global. Dalam penelitian ini, item-item ini <strong>tidak diberikan langsung</strong> ke responden sebagai kuesioner — melainkan diadaptasi menjadi pertanyaan wawancara terbuka yang lebih natural.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DASS-21 — Mengukur Faktor Risiko**")
            dass21_items = {
                "Depresi (D)": [
                    "Saya merasa sulit untuk berinisiatif dalam melakukan sesuatu",
                    "Saya sama sekali tidak merasakan perasaan positif",
                    "Saya merasa bahwa tidak ada hal yang dapat saya harapkan",
                    "Saya merasa sedih dan putus asa",
                    "Saya tidak merasa antusias dalam hal apapun",
                    "Saya merasa bahwa saya tidak berharga sebagai manusia",
                    "Saya merasa bahwa hidup tidak berarti",
                ],
                "Kecemasan (A)": [
                    "Saya merasa bibir saya sering kering",
                    "Saya mengalami kesulitan bernafas",
                    "Saya merasa bahwa saya banyak energi untuk merasa cemas",
                    "Saya merasa khawatir dengan situasi panik",
                    "Saya mengalami gemetar",
                    "Saya menyadari kegiatan jantung yang tidak normal",
                    "Saya merasa takut tanpa alasan yang jelas",
                ],
                "Stres (S)": [
                    "Saya merasa sulit untuk bersistirahat",
                    "Saya cenderung bereaksi berlebihan",
                    "Saya merasa bahwa saya menghabiskan banyak energi karena cemas",
                    "Saya sedang merasa gelisah",
                    "Saya merasa sulit untuk bersantai",
                    "Saya tidak dapat memaklumi hambatan apapun",
                    "Saya merasa bahwa saya mudah tersinggung",
                ],
            }
            for dimensi, items in dass21_items.items():
                with st.expander(f"📌 {dimensi} ({len(items)} item)"):
                    for i, item in enumerate(items, 1):
                        st.write(f"{i}. {item}")
                    st.caption("Skala: 0 = Tidak pernah, 1 = Kadang-kadang, 2 = Sering, 3 = Hampir selalu")

        with col2:
            st.markdown("**MSPSS — Mengukur Faktor Protektif**")
            mspss_items = {
                "Dukungan Keluarga": [
                    "Keluarga saya benar-benar mencoba untuk membantu saya",
                    "Saya mendapatkan bantuan dan dukungan emosional dari keluarga",
                    "Keluarga saya bersedia membantu saya dalam membuat keputusan",
                    "Saya bisa berbicara tentang masalah saya dengan keluarga",
                ],
                "Dukungan Teman": [
                    "Teman-teman saya benar-benar mencoba membantu saya",
                    "Saya bisa berbicara tentang kebahagiaan dan kesedihan dengan teman",
                    "Saya memiliki teman yang bisa berbagi suka duka",
                    "Saya bisa berdiskusi masalah-masalah saya dengan teman",
                ],
                "Significant Others": [
                    "Ada orang khusus yang selalu ada ketika saya membutuhkan",
                    "Ada orang khusus yang peduli pada perasaan saya",
                    "Ada orang khusus yang biasa memberikan kenyamanan",
                    "Ada orang istimewa dalam hidup saya yang peduli",
                ],
            }
            for dimensi, items in mspss_items.items():
                with st.expander(f"📌 {dimensi} ({len(items)} item)"):
                    for i, item in enumerate(items, 1):
                        st.write(f"{i}. {item}")
                    st.caption("Skala: 1=Sangat Tidak Setuju s.d. 7=Sangat Setuju")

    with tab2:
        st.markdown("#### Translasi Instrumen → Pertanyaan Wawancara Naratif")
        st.markdown('<div class="warn-box">⚠️ <strong>Inovasi kunci penelitian ini:</strong> Pertanyaan wawancara didesain agar responden bercerita secara bebas dan natural. Beberapa item DASS-21/MSPSS digabung menjadi satu pertanyaan terbuka untuk efisiensi dan kenyamanan penyintas.</div>', unsafe_allow_html=True)

        mapping = [
            {
                "Pertanyaan Wawancara": "Setelah banjir terjadi, bagaimana perubahan perasaan yang Saudara rasakan dalam seminggu terakhir?",
                "Item DASS-21 yang tercakup": "D3 (tidak ada harapan), D6 (tidak berharga), D7 (hidup tidak berarti)",
                "Dimensi": "Depresi",
                "Contoh jawaban indikator risiko": '"Rasanya hampa, nggak ada gunanya bangun pagi. Tiap hari nangis nggak jelas..."',
            },
            {
                "Pertanyaan Wawancara": "Apakah ada aktivitas yang sebelumnya mudah dilakukan tapi setelah banjir terasa sulit?",
                "Item DASS-21 yang tercakup": "D1 (sulit berinisiatif), D2 (tidak ada afek positif), D5 (tidak antusias)",
                "Dimensi": "Depresi",
                "Contoh jawaban indikator risiko": '"Dulu suka masak buat anak, sekarang nggak ada semangat sama sekali..."',
            },
            {
                "Pertanyaan Wawancara": "Setelah banjir, apakah Saudara pernah merasa khawatir banjir serupa akan terjadi lagi? Situasi apa yang biasanya membuat cemas?",
                "Item DASS-21 yang tercakup": "A4 (khawatir panik), A7 (takut tanpa alasan jelas), A5 (gemetar)",
                "Dimensi": "Kecemasan",
                "Contoh jawaban indikator risiko": '"Setiap hujan lebat, jantung saya berdegup kencang, gemetar, napas sesak..."',
            },
            {
                "Pertanyaan Wawancara": "Saat merasa cemas, apakah tubuh Saudara pernah menunjukkan reaksi fisik tertentu?",
                "Item DASS-21 yang tercakup": "A1 (bibir kering), A2 (sulit bernafas), A6 (detak jantung tidak normal)",
                "Dimensi": "Kecemasan",
                "Contoh jawaban indikator risiko": '"Iya, tangan gemetar, keringat dingin, nafas sesak tiba-tiba..."',
            },
            {
                "Pertanyaan Wawancara": "Hal-hal apa yang saat ini paling membuat Saudara merasa tertekan atau terbebani?",
                "Item DASS-21 yang tercakup": "S1 (sulit istirahat), S3 (banyak energi untuk cemas), S4 (gelisah), S7 (mudah tersinggung)",
                "Dimensi": "Stres",
                "Contoh jawaban indikator risiko": '"Semua saya tanggung sendiri, marah-marah sama anak terus, nyesel sendiri..."',
            },
            {
                "Pertanyaan Wawancara": "Dalam situasi setelah banjir, bagaimana peran keluarga dalam membantu Saudara menjalani hari-hari?",
                "Item MSPSS yang tercakup": "F1 (keluarga mencoba membantu), F2 (dukungan emosional keluarga), F3 (membantu keputusan)",
                "Dimensi": "Protektif — Keluarga",
                "Contoh jawaban indikator risiko": '"Suami jauh, mertua tua, nggak ada yang bisa bantu..."',
            },
            {
                "Pertanyaan Wawancara": "Bisa diceritakan bagaimana hubungan Saudara dengan teman dan tetangga selama masa banjir?",
                "Item MSPSS yang tercakup": "T1 (teman mencoba membantu), T2 (berbagi kebahagiaan/kesedihan), T4 (diskusi masalah)",
                "Dimensi": "Protektif — Teman",
                "Contoh jawaban indikator risiko": '"Di sini kami pendatang, belum banyak kenal orang..."',
            },
            {
                "Pertanyaan Wawancara": "Apakah ada pihak yang sangat berarti bagi Saudara selama masa setelah banjir?",
                "Item MSPSS yang tercakup": "SO1 (ada orang khusus saat dibutuhkan), SO2 (peduli pada perasaan), SO3 (memberi kenyamanan)",
                "Dimensi": "Protektif — Significant Others",
                "Contoh jawaban indikator risiko": '"Tidak ada... semuanya sibuk dengan masalah mereka sendiri..."',
            },
        ]

        df_map = pd.DataFrame(mapping)
        st.dataframe(df_map[["Pertanyaan Wawancara", "Item DASS-21 yang tercakup", "Dimensi"]],
                     use_container_width=True, hide_index=True)

        st.markdown("#### 💡 Contoh Translasi Detail")
        for m in mapping[:3]:
            with st.expander(f"🗣️ \"{m['Pertanyaan Wawancara'][:70]}...\""):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Dimensi:** {m['Dimensi']}")
                    st.markdown(f"**Item yang tercakup:** {m['Item DASS-21 yang tercakup']}")
                with c2:
                    st.markdown(f"**Contoh jawaban indikator risiko tinggi:**")
                    st.markdown(f'<div class="narasi-block">{m["Contoh jawaban indikator risiko"]}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("#### 1.3 Uji Etik & Pilot Study")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card"><h4>🏛️ Ethical Clearance</h4><ul><li>Komite Etik Universitas Syiah Kuala</li><li>Informed consent wajib ditandatangani sebelum wawancara</li><li>Data dianonimisasi (R-001, R-002, dst.)</li><li>Protokol empatik: wawancara dihentikan jika responden distres</li><li>Tersedia rujukan ke layanan psikologis terdekat</li></ul></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><h4>🧪 Pilot Study (5 Responden)</h4><ul><li>Uji kejelasan pertanyaan wawancara</li><li>Estimasi durasi (target 45–90 menit)</li><li>Uji konsistensi antar-rater (Inter-rater Reliability)</li><li>Target Cohen\'s Kappa ≥ 0.75</li><li>Revisi pedoman wawancara jika diperlukan</li></ul></div>', unsafe_allow_html=True)

        st.markdown("#### Ilustrasi Inter-Rater Reliability")
        st.code("""
# Contoh perhitungan Cohen's Kappa antar 2 rater
from sklearn.metrics import cohen_kappa_score

# Rater 1 dan Rater 2 menilai 5 responden pilot
rater1_depresi = ["mild",     "normal",  "severe",   "mild",    "mild"]
rater2_depresi = ["mild",     "normal",  "moderate", "mild",    "mild"]
#                 ✓ agree     ✓ agree    ✗ disagree  ✓ agree    ✓ agree

kappa = cohen_kappa_score(rater1_depresi, rater2_depresi)
print(f"Cohen's Kappa (Depresi): {kappa:.3f}")
# Jika κ < 0.75 → diskusi konsensus → revisi rubrik → ulangi
# Jika κ ≥ 0.75 → rubrik dianggap reliabel → lanjut ke data utama
        """, language="python")


# ─────────────────────────────────────────────
# FASE 2 — PENGUMPULAN DATA
# ─────────────────────────────────────────────
elif fase == "🎤 Fase 2 — Pengumpulan Data":
    st.markdown('<div class="phase-header">🎤 Fase 2 — Pengumpulan Data: Wawancara Naratif</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Di bawah ini adalah <strong>data dummy 5 responden</strong> (dari total 20) yang menggambarkan variasi profil penyintas. Setiap responden memiliki transkrip wawancara naratif lengkap berdasarkan pedoman wawancara yang telah diadaptasi dari DASS-21 dan MSPSS.</div>', unsafe_allow_html=True)

    # Pilih responden
    resp_id = st.selectbox("🔎 Pilih Responden:", list(WAWANCARA_DUMMY.keys()),
                           format_func=lambda x: WAWANCARA_DUMMY[x]["nama"])
    resp = WAWANCARA_DUMMY[resp_id]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👩 Responden", resp_id)
    col2.metric("🎂 Usia", f"{resp['usia']} tahun")
    col3.metric("📍 Lokasi", resp["desa"].split(",")[0])
    col4.metric("⏱️ Durasi", resp["durasi"])

    st.caption(f"📝 Bahasa: {resp['bahasa']}")

    st.divider()

    # Ringkasan label
    lab = resp["label"]
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Depresi", lab["depresi"])
    c2.metric("Kecemasan", lab["kecemasan"])
    c3.metric("Stres", lab["stres"])
    c4.metric("Risiko Overall", lab["risiko_overall"])
    c5.metric("Protektif Keluarga", lab["protektif_keluarga"])
    c6.metric("Protektif Teman", lab["protektif_teman"])
    c7.metric("Protektif Overall", lab["protektif_overall"])

    st.divider()
    st.markdown("#### 📜 Transkrip Wawancara Naratif")

    for sesi in resp["wawancara"]:
        warna = {
            "Depresi": "🔵", "Kecemasan": "🟠", "Stres": "🔴",
            "Rapport": "⚪", "Protektif — Keluarga": "🟢",
            "Protektif — Teman/Komunitas": "🟢", "Protektif — Significant Others": "🟢"
        }
        emoji = warna.get(sesi["dimensi"], "⚪")
        with st.expander(f"{emoji} [{sesi['dimensi']}] {sesi['pertanyaan'][:80]}..."):
            st.markdown(f'<div class="wawancara-q">❓ Pertanyaan:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="wawancara-a">{sesi["pertanyaan"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="wawancara-q" style="margin-top:0.5rem">💬 Jawaban Responden:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="wawancara-a">{sesi["jawaban"]}</div>', unsafe_allow_html=True)

    # Overview semua responden
    st.divider()
    st.markdown("#### 📊 Overview Label Semua Responden (5 dari 20)")
    overview_data = []
    for rid, rdata in WAWANCARA_DUMMY.items():
        lb = rdata["label"]
        overview_data.append({
            "ID": rid, "Usia": rdata["usia"],
            "Depresi": lb["depresi"], "Kecemasan": lb["kecemasan"], "Stres": lb["stres"],
            "Risiko": lb["risiko_overall"],
            "Protektif Keluarga": lb["protektif_keluarga"],
            "Protektif Teman": lb["protektif_teman"],
            "Protektif Overall": lb["protektif_overall"],
        })
    df_ov = pd.DataFrame(overview_data)
    st.dataframe(df_ov, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FASE 3 — PRA-PEMROSESAN
# ─────────────────────────────────────────────
elif fase == "🔧 Fase 3 — Pra-pemrosesan":
    st.markdown('<div class="phase-header">🔧 Fase 3 — Pra-pemrosesan Data: NLP Pipeline Lengkap</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Pilih responden untuk melihat pipeline pra-pemrosesan step-by-step dari teks mentah ke format siap analisis.</div>', unsafe_allow_html=True)

    resp_id = st.selectbox("Pilih Responden:", list(WAWANCARA_DUMMY.keys()),
                           format_func=lambda x: WAWANCARA_DUMMY[x]["nama"])
    resp = WAWANCARA_DUMMY[resp_id]

    # Ambil teks dari jawaban wawancara dimensi depresi
    teks_mentah = resp["wawancara"][1]["jawaban"]  # jawaban depresi sebagai contoh
    teks_bersih = resp["teks_bersih"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "3.1 Transkripsi", "3.2 Normalisasi", "3.3 Tokenisasi",
        "3.4 POS Tagging", "3.5 Stopword & Stemming", "3.6 Output Akhir"
    ])

    with tab1:
        st.markdown("#### 3.1 Transkripsi Verbatim")
        st.markdown('<div class="warn-box"><strong>Prinsip:</strong> Semua kata dipertahankan apa adanya — termasuk filler words ("ee", "umm", "kan"), repetisi, dan kalimat tidak selesai. Ini karena filler words dan repetisi adalah fitur linguistik yang bermakna secara psikologis.</div>', unsafe_allow_html=True)
        st.markdown("**Contoh teks transkripsi mentah:**")
        st.markdown(f'<div class="narasi-block">{teks_mentah}</div>', unsafe_allow_html=True)
        st.markdown("**Fitur linguistik yang perlu dipertahankan:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><strong>Filler words</strong><br>"ee...", "umm...", "ya..." → indikator keraguan & kesulitan bercerita (marker trauma)</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><strong>Repetisi kata</strong><br>"terus-terus", "saya... saya..." → indikator ruminasi & beban kognitif</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><strong>Kalimat tidak selesai</strong><br>Terpotong di tengah → indikasi disosiasi atau emosi terlalu kuat</div>', unsafe_allow_html=True)

        st.markdown("**Alat bantu transkripsi:**")
        st.code("""
# Whisper AI untuk draft transkripsi awal
import whisper
model = whisper.load_model("medium")  # model multilingual
result = model.transcribe("rekaman_R001.mp3", language="id")
print(result["text"])
# Output draft → dikoreksi manual oleh peneliti
# Konvensi: [...]  = bagian tidak terdengar jelas
#           (pause) = jeda panjang
#           [Bahasa Aceh: terjemahan] = kata dalam Bahasa Aceh
        """, language="python")

    with tab2:
        st.markdown("#### 3.2 Normalisasi Teks")
        st.markdown('<div class="info-box"><strong>Perhatian:</strong> Normalisasi dilakukan dengan HATI-HATI — tidak semua kata informal dinormalisasi jika kata aslinya memiliki nilai informatif.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Yang DINORMALISASI:**")
            norm_examples = {
                "nggak / gak / ga": "tidak",
                "gimana": "bagaimana",
                "udah / udeh": "sudah",
                "aja": "saja",
                "buat": "untuk",
                "kayak": "seperti",
                "ngerasa": "merasa",
                "ngurusin": "mengurus",
                "lon (Aceh)": "saya",
                "meunan (Aceh)": "seperti itu",
                "neupeugah (Aceh)": "dikatakan",
                "gata (Aceh)": "kamu",
            }
            df_norm = pd.DataFrame(list(norm_examples.items()), columns=["Bentuk Asli", "Normalisasi"])
            st.dataframe(df_norm, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**⚠️ Yang DIPERTAHANKAN (fitur psikologis):**")
            pertahankan = [
                ("saya", "PRON-I tunggal → marker depresi/ruminasi"),
                ("kami / kita", "PRON-I jamak → marker koneksi sosial"),
                ("ee... / umm...", "Filler word → marker trauma"),
                ("nggak ada", "Negasi kuat → marker hopelessness"),
                ("terus-terus", "Repetisi → marker ruminasi"),
                ("sendirian", "Isolasi sosial → marker risiko"),
                ("doa / Allah / bersyukur", "Kata religius → marker coping protektif"),
            ]
            for asli, alasan in pertahankan:
                st.markdown(f"- **`{asli}`** → {alasan}")

        st.markdown("**Contoh normalisasi pada teks dummy R-001:**")
        st.code(f"""
Input:  "{teks_mentah[:100]}..."

# Proses normalisasi
def normalisasi(teks):
    normalisasi_dict = {{
        'nggak': 'tidak', 'gimana': 'bagaimana', 'udah': 'sudah',
        'ngurusin': 'mengurus', 'ngerasa': 'merasa', 'nyesel': 'menyesal',
        # Kamus Aceh
        'lon': 'saya', 'meunan': 'seperti itu', 'that': 'sangat',
    }}
    for kata_asli, normalisasi in normalisasi_dict.items():
        teks = teks.replace(kata_asli, normalisasi)
    return teks
        """, language="python")

    with tab3:
        st.markdown("#### 3.3 Tokenisasi")
        st.markdown("Memisahkan teks menjadi unit-unit kata (token) dan kalimat yang bisa dianalisis secara individual.")
        st.code("""
from nltk.tokenize import sent_tokenize, word_tokenize

# Tokenisasi kalimat
kalimat_list = sent_tokenize(teks_bersih, language='indonesian')

# Tokenisasi kata per kalimat
for i, kalimat in enumerate(kalimat_list):
    token_kata = word_tokenize(kalimat)
    print(f"Kalimat {i+1}: {token_kata}")
        """, language="python")

        kata_list = teks_bersih.split()
        st.markdown(f"**Output tokenisasi teks bersih {resp_id} ({len(kata_list)} token):**")
        html_tokens = " ".join([f'<span class="token-normal">{k}</span>' for k in kata_list])
        st.markdown(html_tokens, unsafe_allow_html=True)

    with tab4:
        st.markdown("#### 3.4 POS Tagging (Part-of-Speech)")
        st.markdown('<div class="success-box"><strong>Mengapa POS Tagging krusial untuk penelitian ini:</strong><br>• <strong>PRON-SG (saya/aku)</strong> frekuensi tinggi → marker depresi & ruminasi (Trifu et al., 2024)<br>• <strong>PRON-PL (kami/kita)</strong> frekuensi tinggi → marker koneksi sosial, faktor protektif<br>• <strong>ADJ/V-NEG</strong> → skor risiko DASS-21<br>• <strong>NOUN-REL</strong> (kata religius) → faktor protektif koping spiritual khas Aceh</div>', unsafe_allow_html=True)

        st.code("""
# Simulasi POS Tagging (dalam implementasi nyata pakai IndoNLP atau PySastrawi)
# Setiap token diberi label POS sesuai fungsi psikologisnya

Token              POS       Keterangan
──────────────────────────────────────────────────────────
saya               PRON-SG   Kata ganti orang pertama tunggal ← FITUR KUNCI RISIKO
tidak              NEG       Kata negasi
bisa               AUX       Kata bantu
tidur              VERB      Kata kerja
malam              NOUN      Kata benda (konteks temporal)
kepikiran          VERB-COG  Kata kerja kognitif (beban pikiran) ← FITUR KUNCI
terus              ADV       Adverbia frekuensi/intensitas
hampa              ADJ-NEG   Kata sifat afek negatif ← FITUR KUNCI RISIKO
sendirian          ADJ-ISO   Kata sifat isolasi sosial ← FITUR KUNCI RISIKO
bersyukur          ADJ-POS   Kata sifat afek positif ← FITUR KUNCI PROTEKTIF
kami               PRON-PL   Kata ganti orang pertama jamak ← FITUR KUNCI PROTEKTIF
Allah              NOUN-REL  Kata benda religius ← FITUR KUNCI PROTEKTIF
        """, language="text")

        pos_results = pos_tag_dummy(teks_bersih)
        df_pos = pd.DataFrame(pos_results)
        st.dataframe(df_pos, use_container_width=True, hide_index=True,
                     column_config={"POS": st.column_config.TextColumn("POS Tag")})

    with tab5:
        st.markdown("#### 3.5 Stopword Removal (Selektif) & Stemming")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Stopword Removal:**")
            st.markdown('<div class="warn-box">⚠️ <strong>Tidak semua stopword dihapus!</strong> Beberapa stopword adalah fitur psikologis kunci.</div>', unsafe_allow_html=True)
            st.code("""
# Stopword yang DIHAPUS (tidak informatif)
hapus = ['yang', 'adalah', 'dari', 'ke', 'di', 'pada',
         'dengan', 'untuk', 'atau', 'juga', 'ini', 'itu',
         'ada', 'oleh', 'akan', 'dalam', 'sebagai']

# Stopword yang DIPERTAHANKAN (fitur psikologis!)
pertahankan = ['saya', 'aku', 'kami', 'kita',  # pronoun
               'tidak', 'nggak', 'jangan',      # negasi
               'selalu', 'terus', 'sering',      # frekuensi
               'sendirian', 'sendiri',           # isolasi
               'sangat', 'sekali', 'banget']     # intensitas
            """, language="python")

        with col2:
            st.markdown("**Stemming (PySastrawi):**")
            st.code("""
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Contoh stemming
print(stemmer.stem("kepikiran"))      # → pikir
print(stemmer.stem("mengungsi"))      # → ungsi
print(stemmer.stem("ketidakberdayaan"))  # → daya
print(stemmer.stem("kekhawatiran"))   # → khawatir
print(stemmer.stem("menguatkan"))     # → kuat
print(stemmer.stem("bersyukur"))      # → syukur

# CATATAN: Kata religius 'bersyukur' → 'syukur'
# tetap dipertahankan dalam kamus religius
            """, language="python")

    with tab6:
        st.markdown("#### 3.6 Output Akhir: Teks Siap Analisis")
        st.markdown(f"**Teks bersih {resp_id} setelah seluruh pipeline:**")
        st.markdown(f'<div class="narasi-block">{teks_bersih}</div>', unsafe_allow_html=True)

        st.markdown("**Visualisasi token berewarna berdasarkan kategori linguistik:**")
        html_colored = warnai_token(teks_bersih)
        st.markdown(html_colored, unsafe_allow_html=True)

        st.markdown("**Legenda warna:**")
        st.markdown("""
        <span class="token-pron">PRON-I tunggal (risiko)</span>&nbsp;
        <span class="token-afekN">Afek negatif (risiko)</span>&nbsp;
        <span class="token-afekP">Afek positif (protektif)</span>&nbsp;
        <span class="token-sosial">Kata sosial (protektif)</span>&nbsp;
        <span class="token-rel">Kata religius (protektif)</span>&nbsp;
        <span class="token-neg">Negasi</span>&nbsp;
        <span class="token-normal">Lainnya</span>
        """, unsafe_allow_html=True)

        st.markdown("**Format JSON dataset akhir:**")
        st.code(f"""
{{
  "id": "{resp_id}",
  "teks_bersih": "{teks_bersih[:60]}...",
  "tokens": {str(teks_bersih.split()[:8])}...,
  "pos_tags": ["PRON-SG", "NEG", "VERB", "NOUN", "VERB-COG", "ADV", ...],
  "label_risiko": {{
    "depresi": "{resp['label']['depresi']}",
    "kecemasan": "{resp['label']['kecemasan']}",
    "stres": "{resp['label']['stres']}",
    "overall": "{resp['label']['risiko_overall']}"
  }},
  "label_protektif": {{
    "keluarga": "{resp['label']['protektif_keluarga']}",
    "teman": "{resp['label']['protektif_teman']}",
    "significant_other": "{resp['label']['protektif_so']}",
    "overall": "{resp['label']['protektif_overall']}"
  }},
  "metadata": {{
    "durasi_rekaman": "{resp['durasi']}",
    "bahasa_dominan": "{resp['bahasa']}"
  }}
}}
        """, language="json")


# ─────────────────────────────────────────────
# FASE 4 — PELABELAN
# ─────────────────────────────────────────────
elif fase == "🏷️ Fase 4 — Pelabelan Ground Truth":
    st.markdown('<div class="phase-header">🏷️ Fase 4 — Pelabelan Ground Truth oleh Tim Peneliti</div>', unsafe_allow_html=True)
    st.markdown('<div class="warn-box">⚠️ <strong>Inovasi utama:</strong> Tim peneliti (bukan responden) yang melabeli setiap transkrip berdasarkan rubrik DASS-21 & MSPSS. Dua anggota tim membaca transkrip secara independen, kemudian hasil dibandingkan (Inter-rater Reliability). Ini lebih etis untuk penyintas bencana dan menghasilkan data yang lebih kaya naratif.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["4.1 Rubrik DASS-21", "4.2 Rubrik MSPSS", "4.3 IRR", "4.4 Dataset Final"])

    with tab1:
        st.markdown("#### Rubrik Pelabelan DASS-21 (Faktor Risiko)")
        rubrik_dass = {
            "Normal (0)": {
                "Depresi": "Tidak ada ekspresi kesedihan, keputusasaan, atau kehilangan minat. Berbicara tentang aktivitas normal.",
                "Kecemasan": "Tidak ada ekspresi ketakutan, kekhawatiran fisik, atau panik.",
                "Stres": "Tidak ada ekspresi kewalahan, mudah marah berlebihan, atau sulit bersantai.",
            },
            "Mild (1)": {
                "Depresi": "Ada sedikit ekspresi sedih, kurang bersemangat, tapi masih bisa beraktivitas normal.",
                "Kecemasan": "Ada sedikit rasa khawatir, tapi tidak sampai mengganggu fungsi sehari-hari.",
                "Stres": "Merasa agak tertekan/capek, tapi masih bisa diatasi.",
            },
            "Moderate (2)": {
                "Depresi": "Ekspresi sedih/putus asa cukup sering muncul, ada kehilangan minat pada kegiatan yang dulu disenangi.",
                "Kecemasan": "Kekhawatiran cukup intens, ada gejala fisik (jantung berdegup, sulit bernapas).",
                "Stres": "Merasa kewalahan, mudah marah, sulit bersantai meski ingin.",
            },
            "Severe (3)": {
                "Depresi": "Ekspresi kuat tentang tidak ada harapan, tidak berguna, tidak mau beraktivitas.",
                "Kecemasan": "Serangan kecemasan intens, gejala fisik parah, ketakutan sangat kuat.",
                "Stres": "Kewalahan ekstrem, reaksi berlebihan, sulit mengendalikan emosi.",
            },
            "Extremely Severe (4)": {
                "Depresi": "Ekspresi sangat ekstrem tentang tidak ada alasan hidup, kebuntuan total.",
                "Kecemasan": "Panik terus-menerus, ketakutan melumpuhkan fungsi sehari-hari.",
                "Stres": "Tidak bisa berfungsi sama sekali karena stres.",
            },
        }

        for level, items in rubrik_dass.items():
            warna = {"Normal (0)": "🟢", "Mild (1)": "🟡", "Moderate (2)": "🟠", "Severe (3)": "🔴", "Extremely Severe (4)": "🆘"}
            with st.expander(f"{warna.get(level, '⚪')} {level}"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Depresi:** {items['Depresi']}")
                c2.markdown(f"**Kecemasan:** {items['Kecemasan']}")
                c3.markdown(f"**Stres:** {items['Stres']}")

        st.divider()
        st.markdown("#### Contoh Pelabelan dari Transkrip Dummy")
        for rid, rdata in WAWANCARA_DUMMY.items():
            jawaban_depresi = rdata["wawancara"][1]["jawaban"]
            lab = rdata["label"]
            with st.expander(f"📋 {rdata['nama']} — Depresi: {lab['depresi']} | Kecemasan: {lab['kecemasan']} | Stres: {lab['stres']}"):
                st.markdown(f'<div class="narasi-block">{jawaban_depresi}</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Depresi → {lab['depresi']}**")
                c2.markdown(f"**Kecemasan → {lab['kecemasan']}**")
                c3.markdown(f"**Stres → {lab['stres']}**")

    with tab2:
        st.markdown("#### Rubrik Pelabelan MSPSS (Faktor Protektif)")
        rubrik_mspss = {
            "Low Support (1)": {
                "Keluarga": "Keluarga tidak disebut, jauh, atau disebut dengan nada negatif/tidak membantu.",
                "Teman": "Tidak ada teman dekat, komunitas tidak aktif, merasa tidak dikenal.",
                "Significant Others": "Tidak ada sosok khusus yang memberikan dukungan berarti.",
            },
            "Moderate Support (2)": {
                "Keluarga": "Keluarga ada tapi dukungannya terbatas atau tidak konsisten.",
                "Teman": "Ada beberapa teman/tetangga yang membantu secara terbatas.",
                "Significant Others": "Ada satu figur pendukung tapi kehadirannya tidak konsisten atau baru saja pergi.",
            },
            "High Support (3)": {
                "Keluarga": "Keluarga sangat aktif, hadir, dan kompak mendukung pemulihan.",
                "Teman": "Komunitas solid, gotong royong aktif, hubungan sosial kuat.",
                "Significant Others": "Ada tokoh khusus (ustaz, penyuluh, kepala sekolah) yang sangat berperan.",
            },
        }

        for level, items in rubrik_mspss.items():
            warna = {"Low Support (1)": "🔴", "Moderate Support (2)": "🟡", "High Support (3)": "🟢"}
            with st.expander(f"{warna.get(level, '⚪')} {level}"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Keluarga:** {items['Keluarga']}")
                c2.markdown(f"**Teman/Komunitas:** {items['Teman']}")
                c3.markdown(f"**Significant Others:** {items['Significant Others']}")

    with tab3:
        st.markdown("#### 4.3 Inter-Rater Reliability (IRR)")
        st.markdown('<div class="info-box">Dua anggota tim menilai setiap transkrip secara <strong>independen</strong>, kemudian hasil dibandingkan menggunakan Cohen\'s Kappa. Target: κ ≥ 0.75 untuk semua dimensi.</div>', unsafe_allow_html=True)

        irr_data = {
            "Responden": ["R-001", "R-001", "R-002", "R-002", "R-003", "R-003"],
            "Dimensi": ["Depresi", "Kecemasan", "Depresi", "Kecemasan", "Depresi", "Stres"],
            "Rater 1": ["Moderate", "Moderate", "Normal", "Mild", "Severe", "Severe"],
            "Rater 2": ["Moderate", "Severe", "Normal", "Mild", "Severe", "Moderate"],
            "Agree?": ["✅", "❌ → Diskusi", "✅", "✅", "✅", "❌ → Diskusi"],
        }
        st.dataframe(pd.DataFrame(irr_data), use_container_width=True, hide_index=True)

        st.code("""
from sklearn.metrics import cohen_kappa_score

# Setelah semua responden dinilai oleh 2 rater
rater1_semua = ["moderate","moderate","normal","mild","severe","severe", ...]
rater2_semua = ["moderate","severe",  "normal","mild","severe","moderate",...]

kappa = cohen_kappa_score(rater1_semua, rater2_semua)
print(f"Cohen's Kappa overall: {kappa:.3f}")
# Jika κ < 0.75 → diskusi konsensus per kasus tidak setuju
# Jika κ ≥ 0.75 → lanjut analisis
        """, language="python")

    with tab4:
        st.markdown("#### 4.4 Dataset Final Berlabel")
        dataset_rows = []
        for rid, rdata in WAWANCARA_DUMMY.items():
            lab = rdata["label"]
            dataset_rows.append({
                "ID": rid, "Risiko Depresi": lab["depresi"],
                "Risiko Kecemasan": lab["kecemasan"], "Risiko Stres": lab["stres"],
                "Risiko Overall": lab["risiko_overall"],
                "Protektif Keluarga": lab["protektif_keluarga"],
                "Protektif Teman": lab["protektif_teman"],
                "Protektif SO": lab["protektif_so"],
                "Protektif Overall": lab["protektif_overall"],
                "Skor Risiko (1-3)": rdata["skor_risiko_num"],
                "Skor Protektif (1-3)": rdata["skor_protektif_num"],
            })
        df_final = pd.DataFrame(dataset_rows)
        st.dataframe(df_final, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# FASE 5A — EKSPLORASI KEBAHASAAN
# ─────────────────────────────────────────────
elif fase == "🔍 Fase 5A — Eksplorasi Kebahasaan":
    st.markdown('<div class="phase-header">🔍 Fase 5A — Eksplorasi Kebahasaan: LIWC + BERTopic + Spearman</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["5A.1 Analisis LIWC", "5A.2 BERTopic Tema Laten", "5A.3 Korelasi Spearman"])

    with tab1:
        st.markdown("#### 5A.1 Analisis LIWC — Langkah demi Langkah")
        st.markdown('<div class="info-box"><strong>LIWC (Linguistic Inquiry and Word Count)</strong> mengategorikan setiap kata dalam narasi ke dalam kategori psikologis, lalu menghitung frekuensinya sebagai persentase dari total kata. Tim peneliti mengadaptasi kamus LIWC standar ke Bahasa Indonesia + dialek Aceh.</div>', unsafe_allow_html=True)

        st.markdown("**Langkah 1: Bangun kamus kategori psikologis**")
        st.code("""
# Kamus LIWC adaptasi Bahasa Indonesia + Aceh
KAMUS_LIWC = {
    "afek_negatif": [
        "takut", "was-was", "cemas", "sedih", "hampa", "putus asa",
        "nangis", "menyesal", "gemetar", "sesak", "panik", "khawatir",
        "capek", "marah", "tertekan", "tidak berguna", "mimpi buruk",
        # Bahasa Aceh (setelah normalisasi):
        "takot", "susah that", "hana guna",
    ],
    "afek_positif": [
        "bersyukur", "senang", "lega", "tenang", "bangkit", "kuat",
        "semangat", "harapan", "alhamdulillah", "sabar", "ikhlas",
        "kepuasan", "bahagia",
    ],
    "pron_tunggal": ["saya", "aku", "ku", "diri saya", "diri sendiri"],
    "pron_jamak":   ["kami", "kita", "bersama", "bersama-sama"],
    "sosial":       ["keluarga", "suami", "anak", "tetangga", "komunitas",
                     "teman", "saudara", "gotong royong", "saling"],
    "religius":     ["allah", "doa", "berdoa", "ikhlas", "pasrah",
                     "alhamdulillah", "pengajian", "ustaz", "tawakal"],
    "negasi":       ["tidak", "nggak", "tak", "bukan", "belum", "jangan"],
    "isolasi":      ["sendirian", "sendiri", "tidak kenal", "tidak ada yang",
                     "diam-diaman", "jauh", "terpisah"],
}
        """, language="python")

        st.markdown("**Langkah 2: Hitung frekuensi per responden**")
        st.code("""
import re

def hitung_liwc(teks: str, kamus: dict) -> dict:
    teks_lower = teks.lower()
    kata_list = re.findall(r'\\b\\w+\\b', teks_lower)
    total = len(kata_list)
    hasil = {}
    for kategori, kata_kunci in kamus.items():
        count = sum(teks_lower.count(k) for k in kata_kunci)
        hasil[kategori] = round((count / total) * 100, 2)
    return hasil

# Hitung untuk semua responden
df_liwc = pd.DataFrame({rid: hitung_liwc(rdata["teks_bersih"], KAMUS_LIWC)
                         for rid, rdata in data.items()}).T
        """, language="python")

        # Hitung LIWC dummy
        liwc_results = {}
        for rid, rdata in WAWANCARA_DUMMY.items():
            liwc_results[rid] = hitung_liwc(rdata["teks_bersih"])

        df_liwc = pd.DataFrame(liwc_results).T.reset_index()
        df_liwc.columns = ["ID"] + list(df_liwc.columns[1:])

        st.markdown("**Output LIWC per responden (% dari total kata):**")
        st.dataframe(df_liwc, use_container_width=True, hide_index=True)

        st.markdown("**Visualisasi profil linguistik:**")
        fig = go.Figure()
        kategori_tampil = ["afek_negatif", "afek_positif", "pron_tunggal", "pron_jamak", "sosial", "religius"]
        for rid in WAWANCARA_DUMMY.keys():
            vals = [liwc_results[rid].get(k, 0) for k in kategori_tampil]
            fig.add_trace(go.Bar(name=rid, x=kategori_tampil, y=vals))
        fig.update_layout(barmode='group', height=400,
                          title="Perbandingan Profil Linguistik LIWC Antar Responden",
                          xaxis_title="Kategori Linguistik", yaxis_title="% dari Total Kata")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Langkah 3: Interpretasi awal**")
        st.markdown('<div class="success-box">✅ Dari grafik terlihat: R-001 dan R-003 (risiko tinggi) memiliki <strong>afek_negatif dan pron_tunggal yang lebih tinggi</strong>, sementara R-002 dan R-004 (protektif tinggi) memiliki <strong>pron_jamak, sosial, dan religius yang lebih tinggi</strong>. Ini sesuai dengan hipotesis linguistik penelitian.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 5A.2 BERTopic — Pemodelan Tema Laten")
        st.markdown('<div class="info-box"><strong>BERTopic</strong> menggunakan sentence-transformers untuk mengubah setiap narasi menjadi vector embedding, lalu HDBSCAN untuk mengelompokkan narasi ke dalam tema-tema laten. Hasilnya bukan kata tunggal (seperti LIWC) tapi TEMA cerita.</div>', unsafe_allow_html=True)

        st.markdown("**Langkah 1: Setup BERTopic**")
        st.code("""
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# 1. Embedding model — pilih model berbahasa Indonesia
embedding_model = SentenceTransformer("LazarusNLP/all-indo-e5-small-v4")
# Alternatif: "firqaaa/indo-sentence-bert-base"

# 2. Dimensionality reduction
umap_model = UMAP(n_neighbors=5, n_components=3,    # n_neighbors kecil karena N sedikit
                  min_dist=0.0, metric='cosine',
                  random_state=42)

# 3. Clustering
hdbscan_model = HDBSCAN(min_cluster_size=2,         # min_cluster_size=2 karena N=20
                         metric='euclidean',
                         cluster_selection_method='eom',
                         prediction_data=True)

# 4. Rakit BERTopic
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="multilingual",
    calculate_probabilities=True,
    verbose=True
)
        """, language="python")

        st.markdown("**Langkah 2: Fit pada narasi**")
        st.code("""
# Kumpulkan semua narasi (bisa per kalimat atau per responden)
# Untuk N=20, lebih baik split per segmen (per dimensi)
narasi_list = []
id_list = []

for resp_id, resp_data in data.items():
    for sesi in resp_data["wawancara"]:
        if len(sesi["jawaban"]) > 30:  # filter jawaban sangat pendek
            narasi_list.append(sesi["jawaban"])
            id_list.append(f"{resp_id}_{sesi['dimensi']}")

# Fit model
topics, probs = topic_model.fit_transform(narasi_list)

# Lihat info topik
topic_info = topic_model.get_topic_info()
print(topic_info)
        """, language="python")

        st.markdown("**Output topik yang diharapkan muncul:**")
        topik_dummy = [
            {"ID": "Topik 0", "Kata Kunci": ["banjir", "air", "naik", "mengungsi", "rumah", "lumpur"],
             "Label": "Pengalaman Bencana Langsung", "Hipotesis": "Netral — konteks"},
            {"ID": "Topik 1", "Kata Kunci": ["anak", "suami", "keluarga", "sendirian", "ngurusin", "capek"],
             "Label": "Beban Pengasuhan Sendiri", "Hipotesis": "→ Korelasi RISIKO TINGGI"},
            {"ID": "Topik 2", "Kata Kunci": ["takut", "was-was", "malam", "mimpi", "gemetar", "jantung"],
             "Label": "Gejala Kecemasan Somatik", "Hipotesis": "→ Korelasi RISIKO TINGGI"},
            {"ID": "Topik 3", "Kata Kunci": ["tetangga", "bantu", "gotong royong", "kita", "bersama", "solid"],
             "Label": "Dukungan Komunitas", "Hipotesis": "→ Korelasi PROTEKTIF TINGGI"},
            {"ID": "Topik 4", "Kata Kunci": ["Allah", "bersyukur", "doa", "pasrah", "ikhlas", "sabar"],
             "Label": "Coping Religius/Spiritual", "Hipotesis": "→ Korelasi PROTEKTIF TINGGI"},
            {"ID": "Topik 5", "Kata Kunci": ["tidak tahu", "gelap", "mau gimana", "tidak ada harapan"],
             "Label": "Hopelessness & Disorientasi", "Hipotesis": "→ Korelasi RISIKO SANGAT TINGGI"},
        ]
        df_topik = pd.DataFrame(topik_dummy)
        st.dataframe(df_topik, use_container_width=True, hide_index=True)

        st.markdown("**Langkah 3: Visualisasi distribusi topik**")
        fig2 = go.Figure(data=[
            go.Bar(
                x=[t["Label"] for t in topik_dummy],
                y=[8, 5, 6, 7, 9, 3],
                marker_color=['#95a5a6', '#e74c3c', '#e74c3c', '#27ae60', '#27ae60', '#c0392b'],
            )
        ])
        fig2.update_layout(title="Distribusi Frekuensi Topik Narasi (Dummy)",
                           xaxis_title="Topik", yaxis_title="Jumlah Segmen",
                           height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="warn-box"><strong>Catatan penting:</strong> Pada N=20, BERTopic mungkin menghasilkan topik yang kurang stabil. Gunakan <code>min_cluster_size=2</code> dan pertimbangkan menggabungkan topik yang semantically similar secara manual bersama tim psikologi.</div>', unsafe_allow_html=True)

        st.markdown("**Langkah 4: Mapping topik ke responden**")
        st.code("""
# Setiap responden dipetakan ke distribusi topik
# Hitung rata-rata probabilitas topik per responden
import numpy as np

for resp_id in resp_ids:
    idx_resp = [i for i, rid in enumerate(id_list) if rid.startswith(resp_id)]
    prob_resp = probs[idx_resp].mean(axis=0)  # rata-rata probabilitas topik
    print(f"{resp_id}: Topik dominan → {prob_resp.argmax()} (prob={prob_resp.max():.2f})")

# Output contoh:
# R-001: Topik dominan → 2 (Gejala Kecemasan) — prob=0.71
# R-002: Topik dominan → 4 (Coping Religius) — prob=0.68
# R-003: Topik dominan → 5 (Hopelessness) — prob=0.84
        """, language="python")

    with tab3:
        st.markdown("#### 5A.3 Uji Korelasi Spearman")
        st.markdown('<div class="info-box"><strong>Tujuan:</strong> Membuktikan secara statistik fitur linguistik mana yang berkorelasi signifikan dengan label risiko/protektif. Spearman dipilih (bukan Pearson) karena: (1) data ordinal, (2) N kecil, (3) tidak bisa asumsi distribusi normal.</div>', unsafe_allow_html=True)

        st.markdown("**Langkah 1: Siapkan data**")
        st.code("""
import pandas as pd
from scipy.stats import spearmanr

# Gabungkan fitur LIWC dengan label numerik
df = pd.DataFrame({
    'id': ['R-001', 'R-002', 'R-003', 'R-004', 'R-005'],
    'afek_negatif':  [8.2,  3.1, 11.4,  2.8,  5.3],
    'afek_positif':  [2.1,  5.8,  1.3,  6.2,  4.9],
    'pron_tunggal':  [9.4,  4.2, 12.1,  3.1,  5.8],
    'pron_jamak':    [1.2,  7.3,  0.8,  6.8,  4.1],
    'sosial':        [6.3, 11.2,  3.1, 10.8,  7.2],
    'religius':      [4.1,  8.9,  2.2,  9.4,  5.1],
    # Label numerik: 1=Rendah, 2=Sedang, 3=Tinggi
    'skor_risiko':   [3,    1,    3,    2,    2],
    'skor_protektif':[2,    3,    1,    3,    2],
})
        """, language="python")

        st.markdown("**Langkah 2: Hitung korelasi Spearman**")
        st.code("""
fitur_list = ['afek_negatif', 'afek_positif', 'pron_tunggal',
              'pron_jamak', 'sosial', 'religius']

hasil_korelasi = []
for fitur in fitur_list:
    rho_risiko, p_risiko = spearmanr(df[fitur], df['skor_risiko'])
    rho_protektif, p_protektif = spearmanr(df[fitur], df['skor_protektif'])
    hasil_korelasi.append({
        'Fitur': fitur,
        'rho_risiko': round(rho_risiko, 3),
        'p_risiko': round(p_risiko, 4),
        'rho_protektif': round(rho_protektif, 3),
        'p_protektif': round(p_protektif, 4),
        'Signifikan_risiko': '✓' if p_risiko < 0.05 else '✗',
        'Signifikan_protektif': '✓' if p_protektif < 0.05 else '✗',
    })
        """, language="python")

        # Data korelasi dummy
        spearman_data = [
            {"Fitur": "afek_negatif (%)", "rho (vs Risiko)": 0.71, "p-value Risiko": 0.001,
             "rho (vs Protektif)": -0.68, "p-value Protektif": 0.002,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "pron_tunggal (%)", "rho (vs Risiko)": 0.68, "p-value Risiko": 0.002,
             "rho (vs Protektif)": -0.61, "p-value Protektif": 0.008,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "negasi (%)", "rho (vs Risiko)": 0.59, "p-value Risiko": 0.012,
             "rho (vs Protektif)": -0.55, "p-value Protektif": 0.018,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "pron_jamak (%)", "rho (vs Risiko)": -0.65, "p-value Risiko": 0.004,
             "rho (vs Protektif)": 0.72, "p-value Protektif": 0.001,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "sosial (%)", "rho (vs Risiko)": -0.62, "p-value Risiko": 0.006,
             "rho (vs Protektif)": 0.69, "p-value Protektif": 0.002,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "religius (%)", "rho (vs Risiko)": -0.59, "p-value Risiko": 0.009,
             "rho (vs Protektif)": 0.66, "p-value Protektif": 0.004,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "isolasi (%)", "rho (vs Risiko)": 0.51, "p-value Risiko": 0.031,
             "rho (vs Protektif)": -0.48, "p-value Protektif": 0.042,
             "Signifikan Risiko": "✓", "Signifikan Protektif": "✓"},
            {"Fitur": "afek_positif (%)", "rho (vs Risiko)": -0.38, "p-value Risiko": 0.089,
             "rho (vs Protektif)": 0.41, "p-value Protektif": 0.071,
             "Signifikan Risiko": "✗", "Signifikan Protektif": "✗"},
        ]
        df_spearman = pd.DataFrame(spearman_data)
        st.markdown("**Output hasil korelasi Spearman (dummy N=5, ilustrasi):**")
        st.dataframe(df_spearman, use_container_width=True, hide_index=True)

        st.markdown("**Visualisasi kekuatan korelasi:**")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name='Korelasi vs Risiko', x=df_spearman["Fitur"],
            y=df_spearman["rho (vs Risiko)"],
            marker_color=['#e74c3c' if v > 0 else '#27ae60' for v in df_spearman["rho (vs Risiko)"]],
        ))
        fig3.add_hline(y=0.5, line_dash="dash", line_color="orange",
                       annotation_text="Threshold signifikan (ρ=0.5)")
        fig3.add_hline(y=-0.5, line_dash="dash", line_color="orange")
        fig3.update_layout(title="Koefisien Spearman (rho) — Fitur LIWC vs Skor Risiko",
                           yaxis_title="rho", height=380, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="success-box">✅ <strong>Interpretasi:</strong> Fitur dengan |ρ| > 0.5 dan p < 0.05 dianggap signifikan sebagai indikator linguistik. Fitur-fitur ini akan digunakan sebagai input tambahan pada tahap Feature Fusion (Fase 6). Target minimal 3–5 fitur signifikan berhasil dicapai.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FASE 5B — INDOBERT
# ─────────────────────────────────────────────
elif fase == "🤖 Fase 5B — Pemodelan IndoBERT":
    st.markdown('<div class="phase-header">🤖 Fase 5B — Pemodelan Prediktif IndoBERT</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["5B.1 Arsitektur", "5B.2 Strategi N=20", "5B.3 Training", "5B.4 Evaluasi"])

    with tab1:
        st.markdown("#### 5B.1 Arsitektur IndoBERT")
        st.markdown('<div class="info-box"><strong>IndoBERT</strong> (indobenchmark/indobert-base-p1) adalah model BERT yang di-pre-train pada 39GB teks Bahasa Indonesia. Lebih tepat dari mBERT atau XLM-R untuk konteks penelitian ini karena spesifik Indonesia dan mengenal idiom lokal.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Arsitektur IndoBERT:**
            - 12 Transformer layers
            - 768 hidden dimensions
            - 12 attention heads
            - 110M parameters
            - Pre-trained pada corpus Wikipedia Indonesia, news, dan web
            """)
            st.markdown("**Layer yang ditambahkan (fine-tuning):**")
            st.code("""
[CLS] token embedding (768-dim)
    → Dropout (0.1)
    → Linear (768 → 256)
    → ReLU activation
    → Dropout (0.1)
    → Linear (256 → 3)   ← output: RENDAH/SEDANG/TINGGI
    → Softmax
            """, language="text")
        with col2:
            st.code("""
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p1"
)

# Load model dengan classification head
model = BertForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p1",
    num_labels=3,
    # 3 kelas: 0=RENDAH, 1=SEDANG, 2=TINGGI
    problem_type="single_label_classification"
)

# Tokenisasi input
def tokenize_function(batch):
    return tokenizer(
        batch["teks_bersih"],
        truncation=True,
        padding="max_length",
        max_length=256  # narasi tidak terlalu panjang
    )
            """, language="python")

    with tab2:
        st.markdown("#### 5B.2 Strategi untuk Dataset Kecil N=20")
        st.markdown('<div class="danger-box">⚠️ <strong>Tantangan utama:</strong> N=20 sangat kecil untuk fine-tuning model deep learning (biasanya butuh ribuan sampel). Diperlukan beberapa strategi mitigasi.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strategi 1: Segment-Level Training**")
            st.markdown('<div class="card">Alih-alih 1 responden = 1 sampel, gunakan 1 segmen wawancara = 1 sampel.<br><br>• 20 responden × 5 segmen/orang = 100 segmen<br>• Tiap segmen sudah punya label dimensi (depresi/kecemasan/stres)<br>• Efektif meningkatkan data 5× tanpa pengumpulan data baru</div>', unsafe_allow_html=True)
            st.code("""
# Segment-level: setiap dimensi = 1 sampel
for resp_id, resp in data.items():
    for sesi in resp["wawancara"]:
        if sesi["dimensi"] == "Depresi":
            label = resp["label"]["depresi"]
            teks = sesi["jawaban"]
            dataset.append({"teks": teks, "label": label,
                             "id": resp_id, "dimensi": "Depresi"})
            """, language="python")

            st.markdown("**Strategi 2: Leave-One-Out CV (LOO-CV)**")
            st.markdown('<div class="card">Karena N kecil, LOO-CV lebih tepat dari K-Fold biasa.<br><br>• 20 iterasi, tiap iterasi 1 responden = test, 19 lainnya = train<br>• Setiap responden mendapat kesempatan menjadi test set sekali<br>• Hasil lebih reliable daripada 5-Fold CV dengan N=20</div>', unsafe_allow_html=True)
            st.code("""
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
semua_prediksi = []
semua_label = []

for train_idx, test_idx in loo.split(dataset):
    train_data = [dataset[i] for i in train_idx]
    test_data  = [dataset[i] for i in test_idx]
    model.fit(train_data)
    pred = model.predict(test_data)
    semua_prediksi.extend(pred)
    semua_label.extend([d["label"] for d in test_data])
            """, language="python")

        with col2:
            st.markdown("**Strategi 3: Data Augmentation**")
            st.markdown('<div class="card">Memperbanyak data training secara artifisial.</div>', unsafe_allow_html=True)
            st.code("""
# Teknik 1: Back-translation (Indo → Inggris → Indo)
from transformers import pipeline
translator_en = pipeline("translation", model="Helsinki-NLP/opus-mt-id-en")
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")

def back_translate(teks):
    en = translator_en(teks)[0]["translation_text"]
    id_back = translator_id(en)[0]["translation_text"]
    return id_back

# Teknik 2: Sinonim replacement
# Hanya untuk kata NON-fitur-kunci
sinonim = {"sedih": "murung", "takut": "gentar",
           "bersyukur": "berterima kasih"}
            """, language="python")

            st.markdown("**Strategi 4: Freeze & Unfreeze Bertahap**")
            st.markdown('<div class="card">Untuk dataset sangat kecil, jangan langsung fine-tune semua layer — ini menyebabkan catastrophic forgetting.</div>', unsafe_allow_html=True)
            st.code("""
# Tahap 1: Hanya train classification head (epoch 1-3)
for param in model.bert.parameters():
    param.requires_grad = False  # Freeze BERT layers
for param in model.classifier.parameters():
    param.requires_grad = True   # Hanya head yang dilatih

# Tahap 2: Unfreeze 2 layer terakhir (epoch 4-7)
for layer in model.bert.encoder.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Tahap 3: Unfreeze semua (epoch 8-10)
for param in model.bert.parameters():
    param.requires_grad = True
            """, language="python")

    with tab3:
        st.markdown("#### 5B.3 Training Configuration")
        st.code("""
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Definisi metrik evaluasi
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    return {
        "f1_macro": f1_macro,
        "accuracy": accuracy,
    }

# Training arguments — dikonfigurasi untuk dataset kecil
training_args = TrainingArguments(
    output_dir="./indobert-mental-health-aceh",
    num_train_epochs=10,
    per_device_train_batch_size=4,     # batch kecil karena data terbatas
    per_device_eval_batch_size=4,
    learning_rate=2e-5,                # LR standar fine-tuning BERT
    warmup_steps=10,                   # warmup pendek karena data kecil
    weight_decay=0.01,                 # regularisasi mencegah overfitting
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # optimasi untuk F1-Macro
    greater_is_better=True,
    logging_steps=10,
    fp16=False,                        # matikan jika tidak ada GPU
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
        """, language="python")

        st.markdown("**Ilustrasi kurva training (dummy):**")
        epochs = list(range(1, 11))
        train_loss = [1.1, 0.9, 0.75, 0.62, 0.51, 0.43, 0.36, 0.30, 0.25, 0.22]
        val_f1 = [0.41, 0.48, 0.55, 0.61, 0.67, 0.71, 0.73, 0.75, 0.76, 0.77]

        fig_train = make_subplots(specs=[[{"secondary_y": True}]])
        fig_train.add_trace(go.Scatter(x=epochs, y=train_loss, name="Training Loss",
                                       line=dict(color="#e74c3c")), secondary_y=False)
        fig_train.add_trace(go.Scatter(x=epochs, y=val_f1, name="Validation F1-Macro",
                                       line=dict(color="#27ae60")), secondary_y=True)
        fig_train.add_hline(y=0.75, line_dash="dash", line_color="orange",
                            secondary_y=True, annotation_text="Target F1=0.75")
        fig_train.update_layout(title="Kurva Training IndoBERT (Dummy)",
                                xaxis_title="Epoch", height=380)
        st.plotly_chart(fig_train, use_container_width=True)

    with tab4:
        st.markdown("#### 5B.4 Evaluasi Model")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1-Macro (target)", "≥ 0.75", "LOO-CV")
        col2.metric("Accuracy (target)", "≥ 0.80", "LOO-CV")
        col3.metric("AUROC (target)", "≥ 0.80", "per kelas")
        col4.metric("N Iterasi LOO-CV", "20", "satu per responden")

        st.markdown("**Confusion matrix (dummy, ilustrasi):**")
        cm_data = np.array([[6, 1, 0], [1, 4, 1], [0, 1, 6]])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Pred: Rendah", "Pred: Sedang", "Pred: Tinggi"],
            y=["True: Rendah", "True: Sedang", "True: Tinggi"],
            colorscale="Blues", text=cm_data, texttemplate="%{text}",
        ))
        fig_cm.update_layout(title="Confusion Matrix — Klasifikasi Risiko Overall (Dummy N=20)",
                             height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.code("""
from sklearn.metrics import classification_report

print(classification_report(
    y_true, y_pred,
    target_names=["Rendah", "Sedang", "Tinggi"]
))
# Output dummy:
#               precision  recall  f1-score  support
# Rendah         0.86      0.86     0.86       7
# Sedang         0.67      0.67     0.67       6
# Tinggi         0.86      0.86     0.86       7
# macro avg      0.79      0.79     0.79       20  ← F1-Macro tercapai!
        """, language="python")


# ─────────────────────────────────────────────
# FASE 6 — SINTESIS
# ─────────────────────────────────────────────
elif fase == "🗺️ Fase 6 — Sintesis & Pemetaan":
    st.markdown('<div class="phase-header">🗺️ Fase 6 — Sintesis: Integrasi Dua Jalur & Pemetaan Akhir</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["6.1 Mengapa & Bagaimana Gabung", "6.2 Feature Fusion", "6.3 Peta Risiko–Protektif"])

    with tab1:
        st.markdown("#### 6.1 Mengapa Dua Jalur Perlu Digabungkan?")
        st.markdown('<div class="info-box">Pertanyaan yang sering muncul: <em>"BERTopic menghasilkan topik, IndoBERT menghasilkan klasifikasi — ini kan beda hal, kenapa digabung?"</em><br><br>Jawabannya: <strong>Keduanya bukan saingan, tapi saling melengkapi pada level yang berbeda.</strong></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><h4>📊 LIWC</h4><strong>Level:</strong> Kata/token<br><strong>Output:</strong> Frekuensi kategori kata (% dari total)<br><strong>Kekuatan:</strong> Eksplisit, interpretabel, bisa divalidasi psikolog<br><strong>Kelemahan:</strong> Tidak memahami konteks kalimat<br><br>→ <em>"Kata 'tidak' muncul 5 kali"</em></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><h4>🗂️ BERTopic</h4><strong>Level:</strong> Kalimat/paragraf<br><strong>Output:</strong> Tema laten (klaster semantik)<br><strong>Kekuatan:</strong> Memahami makna kontekstual, menemukan tema tak terduga<br><strong>Kelemahan:</strong> Tidak langsung mengklasifikasikan<br><br>→ <em>"Narasi ini bicara tentang beban pengasuhan sendiri"</em></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><h4>🤖 IndoBERT</h4><strong>Level:</strong> Dokumen utuh<br><strong>Output:</strong> Label klasifikasi (Rendah/Sedang/Tinggi)<br><strong>Kekuatan:</strong> Memahami seluruh konteks narasi<br><strong>Kelemahan:</strong> Black box, butuh data banyak<br><br>→ <em>"Responden ini: Risiko TINGGI"</em></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Cara Penggabungan (Feature Fusion)")
        st.markdown("""
        Ada **dua cara** yang bisa dipilih tergantung hasil validasi Spearman:

        **Cara A — Feature Fusion (jika korelasi Spearman signifikan):**
        Fitur LIWC yang terbukti signifikan digabungkan langsung sebagai input tambahan ke classifier IndoBERT.
        Artinya: representasi narasi = [IndoBERT CLS embedding (768 dim)] + [LIWC features (5-7 dim)] = vector 773-775 dim.

        **Cara B — Ensemble/Late Fusion (alternatif):**
        IndoBERT dan model berbasis LIWC (misal: SVM atau Logistic Regression) dijalankan terpisah,
        lalu prediksi akhir diambil berdasarkan majority voting atau weighted average.

        **Catatan penyebutan dalam laporan:**
        Kombinasi ini disebut sebagai **"Pendekatan Hibrida Psikolinguistik"** atau
        **"Model Hybrid LIWC-IndoBERT"** — menegaskan bahwa keduanya berkontribusi
        pada tingkat yang berbeda (leksikal + kontekstual).
        """)

        st.markdown('<div class="success-box">✅ <strong>BERTopic tidak masuk langsung ke feature fusion</strong>, melainkan kontribusinya adalah pada <strong>interpretasi</strong>: setelah model mengklasifikasikan responden, peneliti menggunakan topik BERTopic untuk menjelaskan <em>mengapa</em> seseorang masuk kategori risiko tinggi (karena narasinya didominasi Topik 2: Gejala Kecemasan Somatik, misalnya).</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 6.2 Implementasi Feature Fusion")
        st.code("""
import numpy as np
import torch

def extract_bert_embedding(teks: str, tokenizer, model) -> np.ndarray:
    \"\"\"Ekstrak [CLS] embedding dari IndoBERT (768 dim).\"\"\"
    inputs = tokenizer(teks, return_tensors="pt",
                       truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model.bert(**inputs)
    # [CLS] token adalah token pertama
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding.squeeze()  # shape: (768,)

def buat_fitur_gabungan(teks: str, liwc_dict: dict,
                         tokenizer, model) -> np.ndarray:
    \"\"\"Gabungkan BERT embedding + fitur LIWC yang signifikan.\"\"\"
    # Fitur LIWC yang sudah terbukti signifikan (Spearman)
    fitur_liwc = np.array([
        liwc_dict["afek_negatif"],    # ρ=+0.71 vs risiko
        liwc_dict["pron_tunggal"],    # ρ=+0.68 vs risiko
        liwc_dict["negasi"],          # ρ=+0.59 vs risiko
        liwc_dict["pron_jamak"],      # ρ=-0.65 vs risiko (protektif)
        liwc_dict["sosial"],          # ρ=-0.62 vs risiko (protektif)
        liwc_dict["religius"],        # ρ=-0.59 vs risiko (protektif)
    ])  # shape: (6,)

    bert_emb = extract_bert_embedding(teks, tokenizer, model)  # shape: (768,)
    fused = np.concatenate([bert_emb, fitur_liwc])  # shape: (774,)
    return fused

# Dataset dengan fitur gabungan
X = np.array([buat_fitur_gabungan(d["teks"], d["liwc"], tokenizer, model)
              for d in dataset])
y = np.array([d["label_numerik"] for d in dataset])

# Classifier akhir — MLP sederhana
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=500,
    random_state=42
)

# Evaluasi dengan LOO-CV
from sklearn.model_selection import cross_val_score, LeaveOneOut
scores = cross_val_score(clf, X_scaled, y,
                          cv=LeaveOneOut(),
                          scoring='f1_macro')
print(f"F1-Macro LOO-CV: {scores.mean():.3f} ± {scores.std():.3f}")
        """, language="python")

    with tab3:
        st.markdown("#### 6.3 Peta Risiko–Protektif 2D")
        st.markdown("Setiap responden dipetakan ke dalam scatter plot 2 dimensi: sumbu X = skor risiko, sumbu Y = skor protektif. Posisi dalam kuadran menentukan prioritas intervensi.")

        # Data peta
        peta_data = []
        for rid, rdata in WAWANCARA_DUMMY.items():
            peta_data.append({
                "ID": rid, "Nama": rdata["nama"].split("(")[0].strip(),
                "Skor Risiko": rdata["skor_risiko_num"],
                "Skor Protektif": rdata["skor_protektif_num"],
                "Risiko Overall": rdata["label"]["risiko_overall"],
                "Protektif Overall": rdata["label"]["protektif_overall"],
            })
        df_peta = pd.DataFrame(peta_data)

        warna_map = {"TINGGI": "#e74c3c", "SEDANG": "#f0a500", "RENDAH": "#27ae60"}
        df_peta["Warna"] = df_peta["Risiko Overall"].map(warna_map)

        fig_scatter = go.Figure()
        for _, row in df_peta.iterrows():
            fig_scatter.add_trace(go.Scatter(
                x=[row["Skor Risiko"]], y=[row["Skor Protektif"]],
                mode="markers+text",
                marker=dict(size=20, color=row["Warna"], opacity=0.85),
                text=[row["ID"]], textposition="top center",
                name=row["ID"],
                hovertemplate=f"<b>{row['Nama']}</b><br>Risiko: {row['Risiko Overall']}<br>Protektif: {row['Protektif Overall']}<extra></extra>"
            ))

        # Garis kuadran
        fig_scatter.add_hline(y=2, line_dash="dash", line_color="gray", line_width=1)
        fig_scatter.add_vline(x=2, line_dash="dash", line_color="gray", line_width=1)

        # Label kuadran
        fig_scatter.add_annotation(x=1.2, y=2.8, text="🟢 Resiliensi Kuat", showarrow=False,
                                    font=dict(color="#27ae60", size=11))
        fig_scatter.add_annotation(x=2.8, y=2.8, text="🟡 Berisiko tapi Ada Buffer", showarrow=False,
                                    font=dict(color="#f0a500", size=11))
        fig_scatter.add_annotation(x=1.2, y=1.2, text="⚪ Stabil tapi Rentan", showarrow=False,
                                    font=dict(color="#7f8c8d", size=11))
        fig_scatter.add_annotation(x=2.8, y=1.2, text="🔴 Prioritas Intervensi", showarrow=False,
                                    font=dict(color="#e74c3c", size=11))

        fig_scatter.update_layout(
            title="Peta Risiko–Protektif Perempuan Penyintas Banjir Pidie Jaya",
            xaxis=dict(title="Skor Risiko (1=Rendah, 3=Tinggi)", range=[0.5, 3.5]),
            yaxis=dict(title="Skor Protektif (1=Rendah, 3=Tinggi)", range=[0.5, 3.5]),
            height=500, showlegend=False
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("**Interpretasi kuadran:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="danger-box">🔴 <strong>Risiko Tinggi + Protektif Rendah</strong> (misal: R-003)<br>→ PRIORITAS INTERVENSI SEGERA. Perlu rujukan psikologis profesional dan pendampingan sosial intensif.</div>', unsafe_allow_html=True)
            st.markdown('<div class="warn-box">🟡 <strong>Risiko Tinggi + Protektif Tinggi</strong> (misal: R-001)<br>→ Berisiko namun memiliki buffer. Pantau secara berkala, perkuat jaringan sosial yang sudah ada.</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="success-box">🟢 <strong>Risiko Rendah + Protektif Tinggi</strong> (misal: R-002, R-004)<br>→ Resiliensi kuat. Bisa dijadikan peer support atau model bagi komunitas.</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">⚪ <strong>Risiko Rendah + Protektif Rendah</strong><br>→ Stabil saat ini tapi rentan jika ada stres tambahan. Perlu penguatan jaringan sosial preventif.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FASE 7 — CONTOH HASIL KLASIFIKASI
# ─────────────────────────────────────────────
elif fase == "📄 Fase 7 — Contoh Hasil Klasifikasi":
    st.markdown('<div class="phase-header">📄 Fase 7 — Contoh Lengkap: Wawancara → Klasifikasi Akhir</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Tiga contoh lengkap yang menunjukkan alur end-to-end: dari pertanyaan wawancara yang diadaptasi dari DASS-21 & MSPSS, jawaban naratif responden, sampai ke hasil klasifikasi model dan interpretasi pola linguistik.</div>', unsafe_allow_html=True)

    contoh_ids = ["R-001", "R-002", "R-003"]
    tabs = st.tabs([WAWANCARA_DUMMY[rid]["nama"] for rid in contoh_ids])

    for tab, rid in zip(tabs, contoh_ids):
        with tab:
            resp = WAWANCARA_DUMMY[rid]
            lab = resp["label"]

            # Header
            col1, col2, col3 = st.columns(3)
            with col1:
                warna_r = {"TINGGI": "🔴", "SEDANG": "🟡", "RENDAH": "🟢"}
                st.markdown(f"### {warna_r[lab['risiko_overall']]} Risiko Overall: {lab['risiko_overall']}")
                st.markdown(f"- Depresi: **{lab['depresi']}**")
                st.markdown(f"- Kecemasan: **{lab['kecemasan']}**")
                st.markdown(f"- Stres: **{lab['stres']}**")
            with col2:
                warna_p = {"TINGGI": "🟢", "SEDANG": "🟡", "RENDAH": "🔴"}
                st.markdown(f"### {warna_p[lab['protektif_overall']]} Protektif Overall: {lab['protektif_overall']}")
                st.markdown(f"- Keluarga: **{lab['protektif_keluarga']}**")
                st.markdown(f"- Teman: **{lab['protektif_teman']}**")
                st.markdown(f"- Significant Others: **{lab['protektif_so']}**")
            with col3:
                liwc = hitung_liwc(resp["teks_bersih"])
                st.markdown("### 📊 Profil LIWC")
                st.markdown(f"- Afek Negatif: **{liwc['afek_negatif']}%**")
                st.markdown(f"- PRON Tunggal: **{liwc['pron_tunggal']}%**")
                st.markdown(f"- PRON Jamak: **{liwc['pron_jamak']}%**")
                st.markdown(f"- Sosial: **{liwc['sosial']}%**")
                st.markdown(f"- Religius: **{liwc['religius']}%**")

            st.divider()

            # Wawancara lengkap
            st.markdown("#### 📜 Percakapan Wawancara Lengkap")
            for sesi in resp["wawancara"]:
                emoji_dim = {
                    "Rapport": "🤝", "Depresi": "💙", "Kecemasan": "💛",
                    "Stres": "❤️", "Protektif — Keluarga": "👨‍👩‍👧",
                    "Protektif — Teman/Komunitas": "🤝", "Protektif — Significant Others": "⭐"
                }
                em = emoji_dim.get(sesi["dimensi"], "💬")
                st.markdown(f'<div class="wawancara-q">{em} [{sesi["dimensi"]}] Peneliti: {sesi["pertanyaan"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="wawancara-a">Responden: {sesi["jawaban"]}</div>', unsafe_allow_html=True)

            st.divider()

            # Analisis token
            st.markdown("#### 🎨 Analisis Token Berwarna (Teks Bersih)")
            st.markdown("Warna menunjukkan kategori linguistik yang terdeteksi:")
            html_colored = warnai_token(resp["teks_bersih"])
            st.markdown(f'<div style="background:#f8f9fb;padding:1rem;border-radius:8px;border:1px solid #dde">{html_colored}</div>', unsafe_allow_html=True)
            st.markdown("""
            <span class="token-pron">PRON-I tunggal</span>&nbsp;
            <span class="token-afekN">Afek negatif</span>&nbsp;
            <span class="token-afekP">Afek positif</span>&nbsp;
            <span class="token-sosial">Sosial</span>&nbsp;
            <span class="token-rel">Religius</span>&nbsp;
            <span class="token-neg">Negasi</span>&nbsp;
            <span class="token-normal">Lainnya</span>
            """, unsafe_allow_html=True)

            st.divider()

            # Output model
            st.markdown("#### 🤖 Output Model Klasifikasi (IndoBERT Hybrid)")
            prob_dummy = {
                "R-001": {"Rendah": 0.08, "Sedang": 0.22, "Tinggi": 0.70},
                "R-002": {"Rendah": 0.75, "Sedang": 0.20, "Tinggi": 0.05},
                "R-003": {"Rendah": 0.04, "Sedang": 0.09, "Tinggi": 0.87},
            }
            probs = prob_dummy[rid]
            fig_prob = go.Figure(go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=["#27ae60", "#f0a500", "#e74c3c"],
                text=[f"{v:.0%}" for v in probs.values()],
                textposition="outside",
            ))
            fig_prob.update_layout(title="Probabilitas Output Klasifikasi Risiko",
                                   yaxis_title="Probabilitas", yaxis_range=[0, 1.1],
                                   height=300, showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

            # Penjelasan keputusan
            penjelasan = {
                "R-001": """
                **Mengapa model mengklasifikasikan sebagai RISIKO TINGGI?**
                - PRON tunggal "saya" sangat dominan (9.4%) → marker ruminasi & depresi
                - Afek negatif tinggi (8.2%): "tidak bisa tidur", "capek", "hampa", "sendirian"
                - Negasi berulang: "tidak ada semangat", "tidak ada gunanya"
                - Topik BERTopic dominan: Beban Pengasuhan Sendiri + Gejala Kecemasan
                - Protektif sedang: ada dukungan spiritual tapi keluarga inti jauh
                """,
                "R-002": """
                **Mengapa model mengklasifikasikan sebagai RISIKO RENDAH?**
                - PRON jamak "kami" sangat dominan (7.3%) → koneksi sosial kuat
                - Afek positif dominan (5.8%): "bersyukur", "bangkit", "alhamdulillah"
                - Kata sosial tinggi (11.2%): "gotong royong", "bersama", "solid"
                - Kata religius tinggi (8.9%): "doa", "Allah", "pengajian"
                - Topik BERTopic dominan: Dukungan Komunitas + Coping Religius
                """,
                "R-003": """
                **Mengapa model mengklasifikasikan sebagai RISIKO SANGAT TINGGI?**
                - PRON tunggal ekstrem (12.1%) → ruminasi parah
                - Afek negatif tertinggi (11.4%): "gelap", "tidak ada harapan", "tidak berguna", "gemetar", "sesak"
                - Kata isolasi dominan: "sendirian", "tidak kenal", "diam-diaman"
                - Negasi berulang: "tidak tahu", "tidak bisa", "tidak ada yang"
                - Topik BERTopic dominan: Hopelessness & Disorientasi (prob=0.84)
                - Protektif sangat rendah: keluarga jauh, tidak ada jaringan sosial lokal
                """,
            }
            st.markdown(penjelasan[rid])

            # Rekomendasi
            rekomendasi = {
                "R-001": '<div class="warn-box">⚡ <strong>Rekomendasi:</strong> Perlu pendampingan psikososial berkala. Perkuat jaringan komunitas lokal (kelompok pengajian yang sudah ada). Fasilitasi komunikasi dengan suami/keluarga jauh. Pantau setiap 2 minggu.</div>',
                "R-002": '<div class="success-box">✅ <strong>Rekomendasi:</strong> Resiliensi kuat. Dapat diajak menjadi peer support untuk anggota komunitas lain yang lebih rentan. Pertahankan aktivitas pengajian dan gotong royong yang sudah berjalan.</div>',
                "R-003": '<div class="danger-box">🆘 <strong>Rekomendasi:</strong> PRIORITAS TINGGI. Segera rujuk ke layanan psikologis profesional. Fasilitasi koneksi dengan jejaring sosial lokal (kenalkan dengan komunitas setempat). Pastikan akses ke bantuan ekonomi darurat. Tindak lanjut dalam 3 hari.</div>',
            }
            st.markdown(rekomendasi[rid], unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📌 Ringkasan Pola Linguistik yang Ditemukan")
    pola_data = [
        {"Pola Bahasa", "Interpretasi Psikologis", "Dimensi", "Contoh"},
    ]
    pola_final = [
        {"Pola Bahasa": "PRON-I tunggal frekuensi sangat tinggi (>8%)", "Interpretasi": "Ruminasi & penarikan diri dari lingkungan sosial", "Dimensi": "🔴 Risiko Depresi"},
        {"Pola Bahasa": "Kalimat negasi berulang + kata ketidakberdayaan", "Interpretasi": "Learned helplessness, hopelessness", "Dimensi": "🔴 Risiko Depresi"},
        {"Pola Bahasa": "Kata somatik (jantung berdegup, sesak, gemetar)", "Interpretasi": "Manifestasi fisik kecemasan akut", "Dimensi": "🔴 Risiko Kecemasan"},
        {"Pola Bahasa": "Referensi waktu lampau dominan (kemarin, dulu, sebelum)", "Interpretasi": "Belum berdamai dengan trauma, stuck di past", "Dimensi": "🔴 Risiko PTSD"},
        {"Pola Bahasa": "PRON-I jamak frekuensi tinggi (>5%)", "Interpretasi": "Identifikasi diri dengan komunitas, koneksi sosial", "Dimensi": "🟢 Protektif Sosial"},
        {"Pola Bahasa": "Kata religius (doa, Allah, bersyukur, ikhlas)", "Interpretasi": "Coping spiritual — khas budaya Aceh", "Dimensi": "🟢 Protektif Spiritual"},
        {"Pola Bahasa": "Afiliasi positif + referensi masa depan", "Interpretasi": "Orientasi ke depan, resiliensi aktif", "Dimensi": "🟢 Protektif Resiliensi"},
    ]
    st.dataframe(pd.DataFrame(pola_final), use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption("🧠 Visualisasi Pipeline Penelitian PKM-RSH 2026 | Universitas Syiah Kuala | Cut Alvi Khairanda, Achmad Atha Zayyan, Durratul Irfana | Dosen Pendamping: Dr. Haiyun Nisa, S.Psi., M.Psi., Psikolog")
