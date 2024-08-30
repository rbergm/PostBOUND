SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = c.UserId
  AND c.Score = 2
  AND ph.CreationDate >= CAST('2010-08-19 12:45:55' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-03 21:46:37' AS timestamp)
  AND u.Reputation >= 1
  AND u.Reputation <= 1183
  AND u.Views >= 0;
