import { useState } from 'react';
import { HoneymoonProvider } from './context/HoneymoonContext';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import ItineraryPage from './pages/ItineraryPage';
import BookingsPage from './pages/BookingsPage';
import ScrapbookPage from './pages/ScrapbookPage';
import SettingsPage from './pages/SettingsPage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [filterCity, setFilterCity] = useState(null);

  const goToItinerary = (city = null) => {
    setFilterCity(city);
    setCurrentPage('itinerary');
  };

  const clearFilter = (city = null) => {
    setFilterCity(city);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage setCurrentPage={setCurrentPage} goToItinerary={goToItinerary} />;
      case 'itinerary':
        return <ItineraryPage filterCity={filterCity} clearFilter={clearFilter} />;
      case 'bookings':
        return <BookingsPage />;
      case 'scrapbook':
        return <ScrapbookPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <HomePage setCurrentPage={setCurrentPage} goToItinerary={goToItinerary} />;
    }
  };

  return (
    <HoneymoonProvider>
      <div className="app">
        <Navigation currentPage={currentPage} setCurrentPage={setCurrentPage} />
        <main className="main-content">
          {renderPage()}
        </main>
      </div>
    </HoneymoonProvider>
  );
}

export default App;
