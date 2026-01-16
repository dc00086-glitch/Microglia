import { useHoneymoon } from '../context/HoneymoonContext';
import { MapPin, Calendar, Heart, Plane, Hotel, Camera } from 'lucide-react';
import { format, differenceInDays, parseISO } from 'date-fns';

const cityImages = {
  'Paris': 'https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=400&q=80',
  'Rome': 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=400&q=80',
  'Barcelona': 'https://images.unsplash.com/photo-1583422409516-2895a77efded?w=400&q=80',
  'Santorini': 'https://images.unsplash.com/photo-1613395877344-13d4a8e0d49e?w=400&q=80',
  'Venice': 'https://images.unsplash.com/photo-1514890547357-a9ee288728e0?w=400&q=80',
  'Florence': 'https://images.unsplash.com/photo-1543429258-c5ca3ea2e8a5?w=400&q=80',
  'Amsterdam': 'https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=400&q=80',
  'London': 'https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=400&q=80',
  'Prague': 'https://images.unsplash.com/photo-1519677100203-a0e668c92439?w=400&q=80',
  'Vienna': 'https://images.unsplash.com/photo-1516550893923-42d28e5677af?w=400&q=80',
  'Lisbon': 'https://images.unsplash.com/photo-1585208798174-6cedd86e019a?w=400&q=80',
  'Athens': 'https://images.unsplash.com/photo-1555993539-1732b0258235?w=400&q=80',
  'Dublin': 'https://images.unsplash.com/photo-1549918864-48ac978761a4?w=400&q=80',
  'Munich': 'https://images.unsplash.com/photo-1595867818082-083862f3d630?w=400&q=80',
  'Nice': 'https://images.unsplash.com/photo-1491166617655-0723a0999cfc?w=400&q=80',
  'Amalfi': 'https://images.unsplash.com/photo-1633321702518-7feccafb94d5?w=400&q=80',
  'Cinque Terre': 'https://images.unsplash.com/photo-1516483638261-f4dbaf036963?w=400&q=80',
  'Swiss Alps': 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=400&q=80',
};

const defaultImage = 'https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=400&q=80';

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
        <div className="destination-cards-grid">
          {tripInfo.destinations.map((city, index) => (
            <div key={city} className="destination-card-img" style={{ animationDelay: `${index * 0.1}s` }}>
              <img src={cityImages[city] || defaultImage} alt={city} />
              <div className="destination-overlay">
                <MapPin size={18} />
                <span>{city}</span>
              </div>
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
