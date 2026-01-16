import { useHoneymoon } from '../context/HoneymoonContext';
import { MapPin, Calendar, Heart, Plane, Hotel, Camera } from 'lucide-react';
import { format, differenceInDays, parseISO } from 'date-fns';

export default function HomePage({ setCurrentPage }) {
  const { tripInfo, itinerary, bookings, scrapbook, getTotalBudget } = useHoneymoon();

  const daysUntilTrip = differenceInDays(parseISO(tripInfo.startDate), new Date());
  const tripDuration = differenceInDays(parseISO(tripInfo.endDate), parseISO(tripInfo.startDate)) + 1;
  const totalActivities = itinerary.reduce((sum, day) => sum + day.activities.length, 0);

  return (
    <div className="home-page">
      <header className="hero-section">
        <div className="hero-content">
          <h1>
            <Heart className="heart-icon" fill="#ff6b6b" />
            {tripInfo.couple}'s European Honeymoon
          </h1>
          <p className="hero-subtitle">
            {tripDuration} days of adventure, romance, and unforgettable memories
          </p>
          <div className="trip-dates">
            <Calendar size={18} />
            <span>
              {format(parseISO(tripInfo.startDate), 'MMMM d')} - {format(parseISO(tripInfo.endDate), 'MMMM d, yyyy')}
            </span>
          </div>
          {daysUntilTrip > 0 && (
            <div className="countdown">
              <span className="countdown-number">{daysUntilTrip}</span>
              <span className="countdown-label">days until your adventure begins!</span>
            </div>
          )}
        </div>
      </header>

      <section className="destinations-section">
        <h2>Your Destinations</h2>
        <div className="destination-cards">
          {tripInfo.destinations.map((city, index) => (
            <div key={city} className="destination-card" style={{ animationDelay: `${index * 0.1}s` }}>
              <MapPin size={24} />
              <span>{city}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="stats-section">
        <div className="stat-card" onClick={() => setCurrentPage('itinerary')}>
          <Calendar size={32} />
          <div className="stat-info">
            <span className="stat-number">{itinerary.length}</span>
            <span className="stat-label">Days Planned</span>
          </div>
        </div>
        <div className="stat-card" onClick={() => setCurrentPage('itinerary')}>
          <MapPin size={32} />
          <div className="stat-info">
            <span className="stat-number">{totalActivities}</span>
            <span className="stat-label">Activities</span>
          </div>
        </div>
        <div className="stat-card" onClick={() => setCurrentPage('bookings')}>
          <Plane size={32} />
          <div className="stat-info">
            <span className="stat-number">{bookings.length}</span>
            <span className="stat-label">Bookings</span>
          </div>
        </div>
        <div className="stat-card" onClick={() => setCurrentPage('scrapbook')}>
          <Camera size={32} />
          <div className="stat-info">
            <span className="stat-number">{scrapbook.length}</span>
            <span className="stat-label">Memories</span>
          </div>
        </div>
      </section>

      <section className="budget-section">
        <h2>Trip Budget</h2>
        <div className="budget-card">
          <div className="budget-amount">
            <span className="currency">$</span>
            <span className="amount">{getTotalBudget().toLocaleString()}</span>
          </div>
          <p className="budget-label">Total Booked</p>
          <div className="budget-breakdown">
            <div className="budget-item">
              <Plane size={16} />
              <span>Flights: ${bookings.filter(b => b.type === 'flight').reduce((s, b) => s + (b.cost || 0), 0).toLocaleString()}</span>
            </div>
            <div className="budget-item">
              <Hotel size={16} />
              <span>Hotels: ${bookings.filter(b => b.type === 'hotel').reduce((s, b) => s + (b.cost || 0), 0).toLocaleString()}</span>
            </div>
            <div className="budget-item">
              <MapPin size={16} />
              <span>Activities: ${bookings.filter(b => b.type === 'activity').reduce((s, b) => s + (b.cost || 0), 0).toLocaleString()}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="quick-actions">
        <h2>Quick Actions</h2>
        <div className="action-buttons">
          <button className="action-btn" onClick={() => setCurrentPage('itinerary')}>
            <Calendar size={20} />
            Plan Your Days
          </button>
          <button className="action-btn" onClick={() => setCurrentPage('bookings')}>
            <Plane size={20} />
            Manage Bookings
          </button>
          <button className="action-btn" onClick={() => setCurrentPage('scrapbook')}>
            <Camera size={20} />
            Add Memories
          </button>
        </div>
      </section>
    </div>
  );
}
