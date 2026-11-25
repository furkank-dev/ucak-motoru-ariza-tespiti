// Sayfa mantığı: formu doldurur, backend'e gönderir, sonucu ve turlar tablosunu çizer.

// Her modelin rengi sabit; backend bilinmeyen bir model döndürürse gri olur.
const RENKLER = {
  lojistik: "#257de0", rf: "#f47230", xgboost: "#00b894", svm: "#f2a600",
  mlp: "#ee7fac", rnn: "#008a17", lstm: "#6a4ce2",
};
const renk = (id) => RENKLER[id] || "#9c9c9c";
const DIZI_MODELLER = ["rnn", "lstm"];
const KISA = { lojistik: "Lojistik", rf: "RF", svm: "SVM" };  // tablo başlıkları için

const $ = (id) => document.getElementById(id);
let veri = null;          // veri/veri.json: sensör listesi ve 100 test motoru
let motor = null;         // seçili test motoru
let turlar = JSON.parse(localStorage.getItem("turlar") || "[]");

// "47,23" ya da "47.23" -> 47.23
const sayiOku = (metin) => parseFloat(String(metin).trim().replace(",", "."));
const yaz = (x, basamak) => x.toLocaleString("tr-TR", { useGrouping: false, maximumFractionDigits: basamak ?? 4 });
const durumYaz = (t) => (t ? '<span class="durum ariza">▲ Arızaya yakın</span>' : '<span class="durum saglam">● Sağlam</span>');

// ---------- Girdi formu ----------
function formuKur() {
  $("motor-sec").innerHTML = veri.motorlar.map((m) => `<option value="${m.id}">Motor #${m.id}</option>`).join("");
  $("sensorler").innerHTML = veri.sensorler.map((s) => `
    <div class="sensor" id="kutu-${s.ad}">
      <label for="g-${s.ad}"><b>${s.ad}</b> · ${s.aciklama}${s.birim ? ` (${s.birim})` : ""}</label>
      <input id="g-${s.ad}" data-ad="${s.ad}" inputmode="decimal" autocomplete="off">
      <small id="not-${s.ad}"></small>
    </div>`).join("");
}

function motorSec(id) {
  motor = veri.motorlar.find((m) => m.id === id);
  $("motor-sec").value = id;
  $("motor-bilgi").textContent = `Motor #${motor.id}'in son gözlenen döngüsü (${motor.son_dongu}. döngü). Gerçek durum çalıştırınca açıklanır.`;
  for (const s of veri.sensorler) $(`g-${s.ad}`).value = yaz(motor.sensorler[s.ad]);
  veri.sensorler.forEach(kutuGuncelle);
}

// Değer değiştiyse altın, eğitim aralığının dışındaysa kırmızı işaretle
function kutuGuncelle(s) {
  const deger = sayiOku($(`g-${s.ad}`).value);
  const eski = motor.sensorler[s.ad];
  const kutu = $(`kutu-${s.ad}`), not = $(`not-${s.ad}`);
  const degisti = !Number.isNaN(deger) && Math.abs(deger - eski) > 1e-9;
  const disarda = !Number.isNaN(deger) && (deger < s.min || deger > s.max);
  kutu.classList.toggle("degisti", degisti);
  kutu.classList.toggle("disarda", disarda || Number.isNaN(deger));
  if (Number.isNaN(deger)) not.textContent = "Sayı girin";
  else if (disarda) not.textContent = `Eğitimde görülen aralığın dışında (${yaz(s.min)} – ${yaz(s.max)})`;
  else if (degisti) not.textContent = `Değişti: ${yaz(eski)} → ${yaz(deger)}`;
  else not.textContent = `Eğitimde: ${yaz(s.min)} – ${yaz(s.max)}`;
}

function formuOku() {
  const sensorler = {}, degisiklikler = [];
  for (const s of veri.sensorler) {
    const deger = sayiOku($(`g-${s.ad}`).value);
    if (Number.isNaN(deger)) throw new Error(`${s.ad} için geçerli bir sayı girin.`);
    sensorler[s.ad] = deger;
    if (Math.abs(deger - motor.sensorler[s.ad]) > 1e-9) degisiklikler.push({ ad: s.ad, eski: motor.sensorler[s.ad], yeni: deger });
  }
  return { sensorler, degisiklikler };
}

// ---------- Çalıştırma ----------
async function calistir() {
  const dugme = $("calistir");
  let girdi;
  try { girdi = formuOku(); } catch (h) { $("calistir-not").textContent = h.message; return; }

  dugme.disabled = true;
  dugme.textContent = "Modeller çalışıyor…";
  $("calistir-not").textContent = "";
  try {
    if (!(await baglantiGoster())) throw new Error("Backend'e ulaşılamadı. Backend'in çalıştığından emin olun.");
    const cevap = await tahminIste(motor.id, girdi.sensorler);
    const bilinen = girdi.degisiklikler.length === 0;
    const tur = {
      no: (turlar.at(-1)?.no || 0) + 1,
      motor_id: motor.id,
      degisiklikler: girdi.degisiklikler,
      gercek_etiket: bilinen ? motor.gercek_etiket : null,
      gercek_rul: bilinen ? motor.gercek_rul : null,
      tahminler: cevap.tahminler,
    };
    turlar.push(tur);
    localStorage.setItem("turlar", JSON.stringify(turlar));
    sonucCiz(tur);
    turlarCiz();
  } catch (h) {
    $("sonuc").innerHTML = `<div class="uyari"><b>Çalıştırılamadı.</b> ${h.message}</div>`;
  } finally {
    dugme.disabled = false;
    dugme.textContent = "Tüm modelleri çalıştır";
  }
}

function sonucCiz(tur) {
  const arizaDiyen = tur.tahminler.filter((t) => t.tahmin === 1).length;
  const bilinen = tur.gercek_etiket !== null;
  $("sonuc-baslik").textContent = `Tur ${tur.no} · Motor #${tur.motor_id}`;
  const gercek = bilinen
    ? `<span>Gerçek durum</span><b>${durumYaz(tur.gercek_etiket)}</b><span>Kalan ömür ${tur.gercek_rul} döngü</span>`
    : `<span>Gerçek durum</span><b>Bilinmiyor</b><span>${tur.degisiklikler.length} değer elle değiştirildi</span>`;
  $("sonuc").innerHTML = `
    <div class="ozet">
      <div class="oy"><span>${tur.tahminler.length} modelden</span><b>${arizaDiyen} tanesi “arızaya yakın”</b><span>${tur.tahminler.length - arizaDiyen} tanesi “sağlam” dedi</span></div>
      <div class="gercek">${gercek}</div>
    </div>
    <div class="modeller">${tur.tahminler.map((t) => {
      const dogru = bilinen ? t.tahmin === tur.gercek_etiket : null;
      return `<div class="model" style="border-left-color:${renk(t.id)}">
        <div class="bas"><b>${t.ad}</b><span>${DIZI_MODELLER.includes(t.id) ? "son 30 döngü" : "tek döngü"}</span></div>
        <div class="yuzde sayi">%${yaz(t.olasilik * 100, 1)} <small>arıza olasılığı</small></div>
        <div class="cubuk"><i style="width:${(t.olasilik * 100).toFixed(1)}%"></i></div>
        <div class="alt-satir">${durumYaz(t.tahmin)}${dogru === null ? "" : `<span class="sonuc-isaret ${dogru ? "dogru" : "yanlis"}">${dogru ? "✓ doğru" : "✗ yanlış"}</span>`}</div>
      </div>`;
    }).join("")}</div>
    <p class="dipnot">Karar sınırı %50 (çubuktaki altın çizgi). RNN ve LSTM, motorun önceki döngülerini de görür; ayrıntı API.md'de.</p>`;
}

// ---------- Turlar tablosu ----------
function turlarCiz() {
  const tablo = $("turlar");
  if (!turlar.length) {
    tablo.innerHTML = '<tbody><tr><td class="orta" style="color:var(--gri-yazi);padding:24px">Henüz tur yok.</td></tr></tbody>';
    return;
  }
  // Sütunlar: turlarda görülen bütün modeller, ilk görüldükleri sırayla
  const modeller = [];
  for (const t of turlar) for (const m of t.tahminler) if (!modeller.some((x) => x.id === m.id)) modeller.push({ id: m.id, ad: m.ad });
  const bilinenTurlar = turlar.filter((t) => t.gercek_etiket !== null);

  const satir = (t) => {
    const bilinen = t.gercek_etiket !== null;
    const degisiklik = t.degisiklikler.length
      ? t.degisiklikler.map((d) => `${d.ad}: ${yaz(d.eski)} → ${yaz(d.yeni)}`).join("<br>")
      : '<span style="color:var(--gri-yazi)">yok</span>';
    let dogruSayisi = 0;
    const hucreler = modeller.map((m) => {
      const p = t.tahminler.find((x) => x.id === m.id);
      if (!p) return "<td>—</td>";
      const dogru = bilinen && p.tahmin === t.gercek_etiket;
      if (dogru) dogruSayisi++;
      const isaret = p.tahmin ? '<span class="durum ariza">▲</span>' : '<span class="durum saglam">●</span>';
      return `<td class="orta sayi ${bilinen && !dogru ? "yanlis-hucre" : ""}">${isaret} %${yaz(p.olasilik * 100, 0)}</td>`;
    }).join("");
    return `<tr><td><b>${t.no}</b></td><td>#${t.motor_id}</td><td class="degisiklik">${degisiklik}</td>
      <td>${bilinen ? `${durumYaz(t.gercek_etiket)}<span class="kucuk">RUL ${t.gercek_rul}</span>` : '<span style="color:var(--gri-yazi)">bilinmiyor</span>'}</td>
      ${hucreler}<td class="orta sayi">${bilinen ? `${dogruSayisi} / ${modeller.length}` : "—"}</td></tr>`;
  };

  const toplam = modeller.map((m) => {
    const d = bilinenTurlar.filter((t) => t.tahminler.find((x) => x.id === m.id)?.tahmin === t.gercek_etiket).length;
    return `<td class="orta sayi">${d} / ${bilinenTurlar.length}</td>`;
  }).join("");

  tablo.innerHTML = `
    <thead><tr><th>Tur</th><th>Motor</th><th>Elle değişen</th><th>Gerçek</th>
      ${modeller.map((m) => `<th class="orta"><span class="nokta" style="background:${renk(m.id)}"></span>${KISA[m.id] || m.ad}</th>`).join("")}<th class="orta">Doğru</th></tr></thead>
    <tbody>${[...turlar].reverse().map(satir).join("")}</tbody>
    <tfoot><tr><td colspan="4">Toplam doğru (gerçeği bilinen ${bilinenTurlar.length} tur)</td>${toplam}<td></td></tr></tfoot>`;
  $("turlar-lejant").innerHTML = '<span class="durum ariza">▲</span> arızaya yakın dedi · <span class="durum saglam">●</span> sağlam dedi · yüzde: arıza olasılığı · kırmızı zemin: yanlış karar';
}

// ---------- Başlangıç ----------
async function baglantiGoster() {
  const canli = await backendKontrol();
  const rozet = $("baglanti");
  rozet.classList.toggle("canli", canli);
  rozet.classList.toggle("yok", !canli);
  rozet.querySelector("span").textContent = canli ? "Backend bağlı" : "Backend bağlı değil";
  return canli;
}

async function baslat() {
  veri = await (await fetch("veri/veri.json")).json();
  formuKur();
  motorSec(veri.motorlar[0].id);
  turlarCiz();
  if (turlar.length) sonucCiz(turlar.at(-1));

  $("motor-sec").addEventListener("change", (e) => motorSec(+e.target.value));
  $("rastgele").addEventListener("click", () => motorSec(veri.motorlar[Math.floor(Math.random() * veri.motorlar.length)].id));
  $("sifirla").addEventListener("click", () => motorSec(motor.id));
  $("sensorler").addEventListener("input", (e) => {
    const s = veri.sensorler.find((x) => x.ad === e.target.dataset.ad);
    if (s) kutuGuncelle(s);
  });
  $("calistir").addEventListener("click", calistir);
  $("temizle").addEventListener("click", () => {
    if (!confirm("Bütün turlar silinsin mi?")) return;
    turlar = [];
    localStorage.removeItem("turlar");
    turlarCiz();
    $("sonuc-baslik").textContent = "Modellerin kararı";
    $("sonuc").innerHTML = '<div class="bos">Soldan bir motor seçip <b>Tüm modelleri çalıştır</b>\'a basın.</div>';
  });

  if (!(await baglantiGoster())) {
    $("calistir-not").textContent = "Backend bağlı değil: form doldurulabilir ama modeller backend açılınca çalışır.";
  }
}

baslat();
