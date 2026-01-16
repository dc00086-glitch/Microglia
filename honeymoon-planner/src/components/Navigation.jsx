import { Calendar, Plane, BookHeart, Home, Settings } from 'lucide-react';

export default function Navigation({ currentPage, setCurrentPage }) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'itinerary', label: 'Itinerary', icon: Calendar },
    { id: 'bookings', label: 'Bookings', icon: Plane },
    { id: 'scrapbook', label: 'Scrapbook', icon: BookHeart },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <nav className="navigation">
      <div className="nav-brand">
        <span className="brand-icon">💍</span>
        <span className="brand-text">Our Honeymoon</span>
      </div>
      <ul className="nav-links">
        {navItems.map(item => (
          <li key={item.id}>
            <button
              className={`nav-link ${currentPage === item.id ? 'active' : ''}`}
              onClick={() => setCurrentPage(item.id)}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
