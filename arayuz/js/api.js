// Backend ile konuşan tek dosya. Biçim API.md'de.
// Backend başka bir portta çalışıyorsa burayı değiştirin, örn. "http://localhost:5000/api".
const API_TABAN = "/api";

// Backend ayakta mı? 1,5 sn içinde cevap yoksa bağlı değil sayılır.
async function backendKontrol() {
  try {
    const cevap = await fetch(API_TABAN + "/saglik", { signal: AbortSignal.timeout(1500) });
    return cevap.ok;
  } catch {
    return false;
  }
}

// Formdaki değerleri gönderir, her modelin kararını döndürür.
async function tahminIste(motorId, sensorler) {
  const cevap = await fetch(API_TABAN + "/tahmin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ motor_id: motorId, sensorler }),
  });
  if (!cevap.ok) throw new Error(`Backend hata verdi (${cevap.status}): ${await cevap.text()}`);
  return cevap.json();
}
