SELECT COUNT(*)
FROM postHistory AS ph, posts AS p, users AS u
WHERE ph.PostId = p.Id
  AND p.OwnerUserId = u.Id
  AND ph.CreationDate <= CAST('2014-08-17 21:24:11' AS timestamp)
  AND p.CreationDate >= CAST('2010-07-26 19:26:37' AS timestamp)
  AND p.CreationDate <= CAST('2014-08-22 14:43:39' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 6524
  AND u.Views >= 0;
