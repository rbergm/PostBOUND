SELECT COUNT(*)
FROM postHistory AS ph,
  posts AS p,
  users AS u,
  badges AS b
WHERE b.UserId = u.Id
  AND p.OwnerUserId = u.Id
  AND ph.UserId = u.Id
  AND ph.CreationDate >= CAST('2010-07-19 19:52:31' AS timestamp)
  AND p.Score >= 0
  AND u.CreationDate >= CAST('2010-07-27 02:56:06' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-10 10:44:00' AS timestamp);
