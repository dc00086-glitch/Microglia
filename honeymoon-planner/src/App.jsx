import { useState } from 'react';
import { HoneymoonProvider } from './context/HoneymoonContext';
import Navigation from './components/Navigation';
import HomePage from './pages/HomePage';
import ItineraryPage from './pages/ItineraryPage';
import BookingsPage from './pages/BookingsPage';
import ScrapbookPage from './pages/ScrapbookPage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage setCurrentPage={setCurrentPage} />;
      case 'itinerary':
        return <ItineraryPage />;
      case 'bookings':
        return <BookingsPage />;
      case 'scrapbook':
        return <ScrapbookPage />;
      default:
        return <HomePage setCurrentPage={setCurrentPage} />;
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
