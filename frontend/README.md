# RetailLens Frontend

Responsive React + TypeScript frontend for the RetailLens FastAPI service. The UI uses a light beige and sap-green palette, a slide-out feature menu, and chart/table/split forecast layouts.

## Run locally

Use Node.js 20.19 or newer.

Start the API from the repository root in one terminal:

```powershell
python -m uvicorn api.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally <http://localhost:5173>). The API allows this local origin by default. To use a different API host, create `.env.local` based on `.env.example` and set `VITE_API_BASE_URL`.

The UI reads historical sales, forecast exports, and model reports from the API. Use the `From` and `To` date fields with **Search** to filter historical summaries, records, and chart points. **Reset** restores the full available date range. Forecast output remains a future horizon and is shown separately. Inventory recommendations require the user to enter current stock because the source sales dataset has no inventory values.
